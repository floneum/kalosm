//! Bounded snapshots of already resolved pipeline/grid selections. No tensor,
//! command record or bind group is retained. The normal artifact keys remain
//! authoritative; every dimension read by either the body or grid is checked.
use super::*;

type Built = Vec<Option<(Artifact, [u32; 3], u128)>>;

pub(super) struct WarmArtifacts {
    owner: Option<std::sync::Weak<Plan>>,
    keys: Vec<ArtifactKey>,
    launches: Vec<fusor_ir::extract::Dispatch>,
    buffers: Vec<fusor_ir::extract::BufferPlan>,
    schedules: Vec<Option<fusor_ir::ir::launch::SchedPoint>>,
    dimensions: Vec<(SymId, u64)>,
    built: Arc<Built>,
    symbols: Vec<SymId>,
    scalar_symbols: Vec<SymId>,
    pack: Arc<UniformPack>,
}

impl WarmArtifacts {
    /// At most two snapshots are retained. Charge metadata by bytes rather
    /// than dispatch count; an owned immutable plan needs no structural copy.
    pub(super) fn within_budget(plan: &Plan, binds: &BindingEnv) -> bool {
        use fusor_ir::extract::{BindingPlan, BufferPlan, Dispatch};
        use fusor_ir::ir::launch::SchedPoint;
        use fusor_ir::shape::Dim;
        let mut remaining = 512usize * 1024;
        let mut charge = |count: usize, bytes: usize| {
            count.checked_mul(bytes).and_then(|bytes| {
                remaining = remaining.checked_sub(bytes)?;
                Some(())
            })
        };
        let fits = (|| {
            charge(1, size_of::<Self>())?;
            charge(
                plan.launches.len(),
                size_of::<ArtifactKey>() + size_of::<Option<(Artifact, [u32; 3], u128)>>(),
            )?;
            charge(binds.dims.len(), size_of::<(SymId, u64)>())?;
            // Symbol vectors and the two shared uniform-pack lookup tables.
            // This includes conservative capacity slack for their allocations.
            charge(
                plan.symbols.len().checked_add(plan.scalar_symbols.len())?,
                128,
            )?;
            if !Self::owns(plan, binds) {
                charge(
                    plan.launches.len(),
                    size_of::<Dispatch>() + size_of::<Option<SchedPoint>>(),
                )?;
                charge(plan.buffers.len(), size_of::<BufferPlan>())?;
                for launch in &plan.launches {
                    charge(launch.bindings.len(), size_of::<BindingPlan>())?;
                    charge(launch.members.len(), 2 * size_of::<Id>())?;
                }
                for buffer in &plan.buffers {
                    charge(buffer.layout.shape().len(), 4 * size_of::<Dim>())?;
                }
            }
            Some(())
        })();
        fits.is_some()
    }

    fn owns(plan: &Plan, binds: &BindingEnv) -> bool {
        binds.plan_owner.as_ref().is_some_and(|owner| {
            owner
                .upgrade()
                .is_some_and(|p| std::ptr::eq(p.as_ref(), plan))
        })
    }

    pub(super) fn owned(&self, plan: &Plan, arena: u64, binds: &BindingEnv) -> Option<Arc<Built>> {
        let owner = self.owner.as_ref()?;
        let supplied = binds.plan_owner.as_ref()?;
        (owner.ptr_eq(supplied)
            && supplied
                .upgrade()
                .is_some_and(|p| std::ptr::eq(p.as_ref(), plan))
            && self.keys.first().is_some_and(|k| k.arena == arena)
            && self
                .dimensions
                .iter()
                .all(|(sym, value)| binds.dims.get(sym) == Some(value)))
        .then(|| self.built())
    }

    /// Reuse digests after matching the immutable owner or every structural
    /// input to `plan_artifact_keys`. The weak owner prevents address reuse
    /// and detaches on Arc mutation; a bare pointer or plan hash would not.
    pub(super) fn keys(&self, plan: &Plan, arena: u64) -> Option<Vec<ArtifactKey>> {
        (self.keys.first().is_some_and(|k| k.arena == arena)
            && self.symbols == plan.symbols
            && self.scalar_symbols == plan.scalar_symbols
            && if let Some(owner) = &self.owner {
                std::ptr::eq(owner.as_ptr(), plan)
            } else {
                self.launches == plan.launches
                    && self.buffers == plan.buffers
                    && self
                        .schedules
                        .iter()
                        .zip(&plan.launches)
                        .all(|(schedule, launch)| {
                            *schedule == plan.extraction.theta.get(&launch.root).copied()
                        })
            })
        .then(|| self.keys.clone())
    }

    pub(super) fn uses_arena(&self, arena: u64) -> bool {
        self.keys.iter().any(|key| key.arena == arena)
    }

    pub(super) fn matches(&self, keys: &[ArtifactKey], binds: &BindingEnv) -> bool {
        self.keys == keys
            && self
                .dimensions
                .iter()
                .all(|(sym, value)| binds.dims.get(sym) == Some(value))
    }

    pub(super) fn built(&self) -> Arc<Built> {
        self.built.clone()
    }

    pub(super) fn pack(
        &self,
        arena: u64,
        symbols: &[SymId],
        scalars: &[SymId],
    ) -> Option<Arc<UniformPack>> {
        (self.keys.first().is_some_and(|key| key.arena == arena)
            && self.symbols == symbols
            && self.scalar_symbols == scalars)
            .then(|| self.pack.clone())
    }

    pub(super) fn capture(
        plan: &Plan,
        keys: &[ArtifactKey],
        binds: &BindingEnv,
        built: &Built,
        entries: &lru::LruCache<ArtifactKey, ArtifactEntry>,
        layout: (&[SymId], &[SymId], Arc<UniformPack>),
    ) -> Option<Self> {
        if keys.len() != built.len()
            || keys.len() != plan.launches.len()
            || !Self::within_budget(plan, binds)
        {
            return None;
        }
        let mut dimensions = FxHashMap::default();
        for (key, selected) in keys.iter().zip(built) {
            let (_, _, ph) = selected.as_ref()?;
            let entry = entries.peek(key)?;
            let variant = entry
                .variants
                .peek(&variant_hash(&entry.consulted, binds)?)?;
            if variant.ph != *ph {
                return None;
            }
            for sym in &entry.consulted {
                dimensions.insert(*sym, *binds.dims.get(sym)?);
            }
            if let Some(spec) = &variant.grid_space {
                for dim in &spec.space.dims {
                    dim.evaluate(&mut |sym| {
                        let value = *binds.dims.get(&sym)?;
                        dimensions.insert(sym, value);
                        Some(value)
                    })?;
                }
            }
        }
        let mut dimensions: Vec<_> = dimensions.into_iter().collect();
        dimensions.sort_unstable_by_key(|(sym, _)| *sym);
        let owned = Self::owns(plan, binds);
        Some(Self {
            owner: owned.then(|| binds.plan_owner.clone()).flatten(),
            keys: keys.to_vec(),
            launches: if owned {
                Vec::new()
            } else {
                plan.launches.clone()
            },
            buffers: if owned {
                Vec::new()
            } else {
                plan.buffers.clone()
            },
            schedules: if owned {
                Vec::new()
            } else {
                plan.launches
                    .iter()
                    .map(|l| plan.extraction.theta.get(&l.root).copied())
                    .collect()
            },
            dimensions,
            built: Arc::new(built.clone()),
            symbols: layout.0.to_vec(),
            scalar_symbols: layout.1.to_vec(),
            pack: layout.2,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_ir::{ir::launch::IndexSpace, shape::Dim};

    #[test]
    fn snapshot_checks_body_grid_and_schedule_but_not_runtime_scalars() {
        let body = SymId(17);
        let grid = SymId(23);
        let scalar = SymId(29);
        let key = ArtifactKey {
            arena: 1,
            launch: 2,
            context: 3,
        };
        let binds = BindingEnv::default()
            .with_dim(body, 128)
            .with_dim(grid, 17)
            .with_scalar(scalar, 1.0);
        let artifact = Artifact::new(42u32);
        let mut variants = lru::LruCache::new(NonZeroUsize::new(2).unwrap());
        variants.put(
            variant_hash(&[body], &binds).unwrap(),
            ArtifactVariant {
                artifact: artifact.clone(),
                grid: [1, 1, 1],
                ph: 42,
                grid_space: Some(crate::lower::GridSpec {
                    space: IndexSpace::new([Dim::Sym(grid)]),
                    block: 32,
                    inner_tile: 1,
                }),
            },
        );
        let mut entries = lru::LruCache::new(NonZeroUsize::new(2).unwrap());
        entries.put(
            key,
            ArtifactEntry {
                consulted: vec![body],
                variants,
            },
        );
        let built = vec![Some((artifact, [1, 1, 1], 42))];
        let symbols = [body, grid, scalar];
        let scalars = [scalar];
        let plan = Plan {
            extraction: Default::default(),
            launches: vec![fusor_ir::extract::Dispatch {
                root: Id(1),
                members: smallvec::smallvec![Id(1)],
                bindings: vec![fusor_ir::extract::BindingPlan {
                    binding: 1,
                    value: Id(1),
                    kind: fusor_ir::extract::BindKind::Write,
                    arena: false,
                }],
                grid: [1, 1, 1],
                block: 32,
            }],
            buffers: vec![fusor_ir::extract::BufferPlan {
                value: Id(1),
                layout: fusor_ir::shape::Layout::contiguous(&[Dim::Const(32)]),
                elements: Dim::Const(32),
                dtype: fusor_ir::dtype::Dtype::F32,
                persistence: fusor_ir::dtype::Persistence::Step,
                arena: None,
            }],
            arena_bytes: 0,
            symbols: symbols.to_vec(),
            scalar_symbols: scalars.to_vec(),
            hash: Default::default(),
            cost: fusor_ir::cost::Picoseconds(0),
        };
        let layout = || (&symbols[..], &scalars[..], Arc::new(UniformPack::default()));
        let snapshot =
            WarmArtifacts::capture(&plan, &[key], &binds, &built, &entries, layout()).unwrap();
        assert!(snapshot.keys(&plan, 1).is_some());
        assert!(snapshot.keys(&plan, 2).is_none());
        let mut changed = plan.clone();
        changed.launches[0].bindings[0].binding = 2;
        assert!(snapshot.keys(&changed, 1).is_none());
        let mut changed = plan.clone();
        changed.buffers[0].layout = fusor_ir::shape::Layout::contiguous(&[Dim::Const(64)]);
        assert!(snapshot.keys(&changed, 1).is_none());
        let mut changed = plan.clone();
        changed
            .extraction
            .theta
            .insert(Id(1), fusor_ir::ir::launch::SchedPoint::Point);
        assert!(snapshot.keys(&changed, 1).is_none());
        let mut changed = plan.clone();
        changed.symbols.swap(0, 1);
        assert!(snapshot.keys(&changed, 1).is_none());
        assert!(snapshot.matches(&[key], &binds));
        assert!(snapshot.matches(&[key], &binds.clone().with_scalar(scalar, 9.0)));
        assert!(!snapshot.matches(&[key], &binds.clone().with_dim(body, 256)));
        assert!(!snapshot.matches(&[key], &binds.clone().with_dim(grid, 33)));
        assert!(!snapshot.matches(&[ArtifactKey { context: 4, ..key }], &binds));
        let mut missing = binds.clone();
        missing.dims.remove(&grid);
        assert!(!snapshot.matches(&[key], &missing));
        assert!(
            WarmArtifacts::capture(&plan, &[key], &missing, &built, &entries, layout()).is_none()
        );
        assert!(snapshot.pack(1, &[body, grid, scalar], &[scalar]).is_some());
        assert!(snapshot.pack(2, &[body, grid, scalar], &[scalar]).is_none());
        assert!(snapshot.pack(1, &[grid, body, scalar], &[scalar]).is_none());
        assert!(snapshot.pack(1, &[body, grid, scalar], &[body]).is_none());

        let mut owned = Arc::new(plan.clone());
        let mut owner_binds = binds.clone();
        owner_binds.plan_owner = Some(Arc::downgrade(&owned));
        let owned_snapshot =
            WarmArtifacts::capture(&owned, &[key], &owner_binds, &built, &entries, layout())
                .unwrap();
        assert!(owned_snapshot.owned(&owned, 1, &owner_binds).is_some());
        assert!(owned_snapshot.keys(&owned, 1).is_some());
        assert!(owned_snapshot.keys(&plan, 1).is_none());
        assert!(owned_snapshot.launches.is_empty());
        assert!(owned_snapshot.buffers.is_empty());
        assert!(owned_snapshot.schedules.is_empty());
        assert!(owned_snapshot.owned(&plan, 1, &owner_binds).is_none());
        assert!(owned_snapshot.owned(&owned, 2, &owner_binds).is_none());
        assert!(
            owned_snapshot
                .owned(&owned, 1, &owner_binds.clone().with_dim(grid, 33))
                .is_none()
        );
        assert!(
            owned_snapshot
                .owned(&owned, 1, &owner_binds.clone().with_scalar(scalar, 4.0))
                .is_some()
        );
        assert_eq!(Arc::strong_count(&owned), 1);
        Arc::make_mut(&mut owned).launches[0].block = 64;
        owner_binds.plan_owner = Some(Arc::downgrade(&owned));
        assert!(owned_snapshot.owned(&owned, 1, &owner_binds).is_none());
        assert!(owned_snapshot.keys(&owned, 1).is_none());
        // A caller's hint can name a different plan. Capture must not
        // associate that identity with the borrowed plan's artifacts.
        let mislabelled =
            WarmArtifacts::capture(&plan, &[key], &owner_binds, &built, &entries, layout())
                .unwrap();
        assert!(mislabelled.owned(&owned, 1, &owner_binds).is_none());

        // A sparse plan beyond the former dispatch-count cutoff still fits
        // the byte bound. Its weak owner avoids retaining the full structure.
        let mut large = plan.clone();
        large.launches.resize(1024, plan.launches[0].clone());
        let mut large = Arc::new(large);
        let mut large_binds = binds.clone();
        large_binds.plan_owner = Some(Arc::downgrade(&large));
        let large_keys = vec![key; large.launches.len()];
        let large_built = vec![built[0].clone(); large.launches.len()];
        let snapshot = WarmArtifacts::capture(
            &large,
            &large_keys,
            &large_binds,
            &large_built,
            &entries,
            layout(),
        )
        .unwrap();
        assert!(snapshot.owned(&large, 1, &large_binds).is_some());
        assert!(snapshot.launches.is_empty());
        assert_eq!(Arc::strong_count(&large), 1);
        Arc::make_mut(&mut large).symbols.resize(8192, body);
        large_binds.plan_owner = Some(Arc::downgrade(&large));
        assert!(!WarmArtifacts::within_budget(&large, &large_binds));
        assert!(
            WarmArtifacts::capture(
                &large,
                &large_keys,
                &large_binds,
                &large_built,
                &entries,
                layout(),
            )
            .is_none()
        );
    }
}
