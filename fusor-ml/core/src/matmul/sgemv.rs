use crate::MatMulOperation;

pub(crate) fn dispatch_size(
    m: u32,
    n: u32,
    batch_size: u32,
    _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
    params: &SgemvParams,
) -> [u32; 3] {
    let total = m.div_ceil(params.chunk_size) * n * batch_size;
    let max = 65_535;
    let x = total.min(max);
    let y = total.div_ceil(x).min(max);
    let z = total.div_ceil(x * y);
    [x, y, z]
}

pub(crate) fn workgroup_shape_constraints(
    _: &MatMulOperation,
    device: &crate::Device,
    params: &SgemvParams,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::less_than(
            device.limits().max_compute_workgroup_size_x + 1,
        ),
    );
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::more_than_or_equals(device.min_subgroup_size()),
    );
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::less_than_or_equals(
            device.max_subgroup_size()
                * params
                    .subgroups_per_workgroup
                    .min(device.max_subgroup_size()),
        ),
    );
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}

#[derive(Debug, Clone, Hash)]
pub struct SgemvParams {
    chunk_size: u32,
    vector_size: u32,
    subgroups_per_workgroup: u32,
}

impl SgemvParams {
    pub fn new(chunk_size: u32, vector_size: u32, subgroups_per_workgroup: u32) -> Self {
        Self {
            chunk_size,
            vector_size,
            subgroups_per_workgroup,
        }
    }

    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    pub fn vector_size(&self) -> u32 {
        self.vector_size
    }

    pub fn subgroups_per_workgroup(&self) -> u32 {
        self.subgroups_per_workgroup
    }
}

impl Default for SgemvParams {
    fn default() -> Self {
        SgemvParams::new(1, 4, 1)
    }
}
