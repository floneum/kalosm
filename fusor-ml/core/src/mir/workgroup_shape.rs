// The shape of the workgroup. [x, y, z] where their product is bounded by
// the device's workgroup limits.
//
// Kernels can be fused if their workgroup shape can be coerced. Coercion can happen if
// the biggest linearized workgroup shape is a multiple of all smaller workgroup shapes.

use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::FxBuildHasher;
use std::{num::NonZeroUsize, sync::OnceLock};

#[derive(Debug, Clone, Copy)]
pub struct WorkgroupShape {
    shape: [u32; 3],
}

impl From<[u32; 3]> for WorkgroupShape {
    fn from(shape: [u32; 3]) -> Self {
        Self { shape }
    }
}

impl WorkgroupShape {
    pub(crate) fn new(x: u32, y: u32, z: u32) -> Self {
        assert!(
            x > 0 && y > 0 && z > 0,
            "Workgroup shape dimensions must be greater than zero"
        );
        Self { shape: [x, y, z] }
    }

    pub(crate) fn linearized(&self) -> u32 {
        self.shape.iter().product()
    }

    pub(crate) fn x(&self) -> u32 {
        self.shape[0]
    }

    pub(crate) fn y(&self) -> u32 {
        self.shape[1]
    }

    pub(crate) fn z(&self) -> u32 {
        self.shape[2]
    }

    pub(crate) fn shape(&self) -> [u32; 3] {
        self.shape
    }
}

impl IntoIterator for WorkgroupShape {
    type Item = u32;
    type IntoIter = std::array::IntoIter<u32, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.shape.into_iter()
    }
}

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct WorkgroupShapeConstraints {
    shape: [Vec<Constraint>; 3],
}

impl WorkgroupShapeConstraints {
    #[track_caller]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn add_constraint(&mut self, dimension: usize, constraint: Constraint) {
        assert!(dimension < 3, "Dimension must be 0, 1, or 2");
        self.shape[dimension].push(constraint);
    }

    fn is_valid(&self, shape: &WorkgroupShape) -> bool {
        self.shape.iter().enumerate().all(|(i, constraints)| {
            constraints
                .iter()
                .all(|constraint| constraint.fits(shape.shape[i]))
        })
    }

    fn possible(&self, limits: &wgpu::Limits) -> impl Iterator<Item = WorkgroupShape> {
        possible_workgroup_shapes(limits).filter(move |shape| self.is_valid(shape))
    }

    pub(crate) fn solve(
        &self,
        max_subgroup_size: u32,
        limits: &wgpu::Limits,
    ) -> Option<WorkgroupShape> {
        type CacheKey = (WorkgroupShapeConstraints, u32, [u32; 4]);
        static CACHE: OnceLock<RwLock<LruCache<CacheKey, Option<WorkgroupShape>, FxBuildHasher>>> =
            OnceLock::new();
        let cache = CACHE.get_or_init(|| {
            RwLock::new(LruCache::with_hasher(
                const { NonZeroUsize::new(2048).unwrap() },
                Default::default(),
            ))
        });
        let key = (
            self.clone(),
            max_subgroup_size,
            [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
                limits.max_compute_invocations_per_workgroup,
            ],
        );
        let mut write = cache.write();
        *write.get_or_insert_ref(&key, || {
            // Find the smallest valid shape that matches the max subgroup size
            self.possible(limits).min_by_key(|shape| {
                let linearized = shape.linearized();
                (linearized as i64)
                    + if max_subgroup_size == 0 || shape.x() % max_subgroup_size == 0 {
                        0
                    } else {
                        1024
                    }
            })
        })
    }
}

/// Every workgroup shape the device can run: each dimension within its
/// per-axis limit and the product within the invocation limit.
fn possible_workgroup_shapes(limits: &wgpu::Limits) -> impl Iterator<Item = WorkgroupShape> {
    let total = limits.max_compute_invocations_per_workgroup;
    let max_x = limits.max_compute_workgroup_size_x.min(total);
    let max_y = limits.max_compute_workgroup_size_y.min(total);
    let max_z = limits.max_compute_workgroup_size_z.min(total);
    (1..=max_x).flat_map(move |x| {
        (1..=max_y.min(total / x)).flat_map(move |y| {
            (1..=max_z.min(total / (x * y))).map(move |z| WorkgroupShape::new(x, y, z))
        })
    })
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) enum Constraint {
    Equals(u32),
    LessThan(u32),
    Not(Box<Constraint>),
}

impl Constraint {
    pub(crate) fn equals(value: u32) -> Self {
        Constraint::Equals(value)
    }

    pub(crate) fn less_than(value: u32) -> Self {
        Constraint::LessThan(value)
    }

    pub(crate) fn more_than_or_equals(value: u32) -> Self {
        Constraint::Not(Box::new(Constraint::LessThan(value)))
    }

    pub(crate) fn less_than_or_equals(value: u32) -> Self {
        Constraint::LessThan(value + 1)
    }

    fn fits(&self, shape: u32) -> bool {
        match self {
            Constraint::Equals(value) => shape == *value,
            Constraint::LessThan(value) => shape < *value,
            Constraint::Not(inner) => !inner.fits(shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Constraint, WorkgroupShapeConstraints, possible_workgroup_shapes};

    const TEST_MAX_SUBGROUP_SIZE: u32 = 64;

    fn test_limits(size: u32) -> wgpu::Limits {
        wgpu::Limits {
            max_compute_workgroup_size_x: size,
            max_compute_workgroup_size_y: size,
            max_compute_workgroup_size_z: size,
            max_compute_invocations_per_workgroup: size,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn test_all_possible_workgroup_shapes() {
        assert_eq!(possible_workgroup_shapes(&test_limits(256)).count(), 5136);
        assert_eq!(possible_workgroup_shapes(&test_limits(1024)).count(), 30343);
    }

    #[test]
    fn test_workgroup_shape_constraints() {
        let limits = test_limits(256);
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::Equals(4));
        constraints.add_constraint(1, Constraint::LessThan(3));

        for shape in constraints.possible(&limits) {
            assert_eq!(shape.shape()[0], 4);
            assert!(shape.shape()[1] < 3);
        }

        let valid_shape = constraints.solve(TEST_MAX_SUBGROUP_SIZE, &limits).unwrap();
        assert_eq!(valid_shape.shape(), [4, 1, 1]);
        assert_eq!(valid_shape.linearized(), 4);
    }
}
