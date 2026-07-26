use crate::{
    QMatrix, TensorData,
    mir::{
        inputs::MirValue,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
};

#[derive(Clone, Debug)]
pub(crate) enum MaybeQData {
    Tensor(TensorData),
    QMatrix(QMatrix),
}

impl MaybeQData {
    pub(crate) fn device(&self) -> &crate::Device {
        match self {
            MaybeQData::Tensor(tensor) => tensor.device(),
            MaybeQData::QMatrix(qmatrix) => qmatrix.device(),
        }
    }
}

impl From<TensorData> for MaybeQData {
    fn from(tensor: TensorData) -> Self {
        Self::Tensor(tensor)
    }
}

impl From<&TensorData> for MaybeQData {
    fn from(tensor: &TensorData) -> Self {
        Self::Tensor(tensor.clone())
    }
}

impl From<QMatrix> for MaybeQData {
    fn from(qmatrix: QMatrix) -> Self {
        Self::QMatrix(qmatrix)
    }
}

impl From<&QMatrix> for MaybeQData {
    fn from(qmatrix: &QMatrix) -> Self {
        Self::QMatrix(qmatrix.clone())
    }
}

impl From<MaybeQData> for MirValue {
    fn from(val: MaybeQData) -> Self {
        match val {
            MaybeQData::Tensor(tensor) => MirValue::Tensor(tensor),
            MaybeQData::QMatrix(qmatrix) => MirValue::QMatrix(qmatrix),
        }
    }
}

impl TryFrom<MirValue> for MaybeQData {
    type Error = ();

    fn try_from(value: MirValue) -> Result<Self, Self::Error> {
        match value {
            MirValue::Tensor(tensor) => Ok(MaybeQData::Tensor(tensor)),
            MirValue::QMatrix(qmatrix) => Ok(MaybeQData::QMatrix(qmatrix)),
            _ => Err(()),
        }
    }
}

pub(crate) fn titled_map_workgroup_size_constraints(
    _shape: &[usize],
    device: &crate::Device,
) -> WorkgroupShapeConstraints {
    let mut constraints = WorkgroupShapeConstraints::new();
    let workgroup_size = device.limits().max_compute_workgroup_size_x.min(256);

    constraints.add_constraint(0, Constraint::equals(workgroup_size));
    constraints.add_constraint(1, Constraint::equals(1));
    constraints.add_constraint(2, Constraint::equals(1));

    constraints
}

pub(crate) fn distribute_workgroups(total_workgroups: u32, max_per_dim: u32) -> [u32; 3] {
    let max_per_dim = max_per_dim.max(1);
    if total_workgroups <= max_per_dim {
        return [total_workgroups, 1, 1];
    }
    // Pick the slab count first, then size X to the slab. Saturating X instead
    // leaves the last slab nearly empty (122,880 groups would launch
    // [65535, 2, 1] = 131,070), and every extra workgroup still runs the
    // kernel prologue and the in-range compares before falling through.
    let y = total_workgroups.div_ceil(max_per_dim).min(max_per_dim);
    let x = total_workgroups.div_ceil(y).min(max_per_dim);
    let z = total_workgroups.div_ceil(x.saturating_mul(y)).max(1);

    [x, y, z]
}

pub(crate) fn titled_map_dispatch_size(
    tile_size: u32,
    workgroup_shape: WorkgroupShape,
    shape: &[usize],
    max_per_dim: u32,
) -> [u32; 3] {
    let total_elements: u64 = shape.iter().map(|&x| x as u64).product();
    let total_tiles = total_elements.div_ceil(tile_size as u64) as u32;
    let workgroup_volume = workgroup_shape.x() * workgroup_shape.y() * workgroup_shape.z();
    let total_workgroups = total_tiles.div_ceil(workgroup_volume);

    distribute_workgroups(total_workgroups, max_per_dim)
}

#[test]
fn distribute_workgroups_covers_without_slack() {
    const MAX: u32 = 0xFFFF;
    for total in (0..3_000_000)
        .step_by(1409)
        .chain([0, 1, MAX, MAX + 1, 122_880, u32::MAX])
    {
        let [x, y, z] = distribute_workgroups(total, MAX);
        assert!(
            x <= MAX && y <= MAX && z <= MAX,
            "{total} exceeds the limit"
        );
        let launched = x as u64 * y as u64 * z as u64;
        assert!(launched >= total as u64, "{total} is not covered");
        // Every launched group past `total` still runs the kernel prologue, so
        // the slack must stay below one slab rather than up to a full X extent.
        assert!(
            total > MAX * MAX || (launched - total as u64) < y as u64,
            "{total} launches {launched}"
        );
    }
}
