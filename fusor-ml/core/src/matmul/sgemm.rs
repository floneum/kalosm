use crate::MatMulOperation;

pub(super) fn workgroup_shape_constraints(
    _: &MatMulOperation,
    _: &crate::Device,
    parameters: &SgemmParams,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::Equals(
            (parameters.block_m_size * parameters.block_n_size)
                / (parameters.thread_m_size * parameters.thread_n_size),
        ),
    );
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}

pub(super) fn dispatch_size(
    last_dim_size: usize,
    second_to_last_dim_size: usize,
    batch_size: usize,
    workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
    parameters: &SgemmParams,
) -> [u32; 3] {
    [
        (last_dim_size as u32).div_ceil(parameters.block_n_size),
        (second_to_last_dim_size as u32).div_ceil(parameters.block_m_size),
        (batch_size as u32).div_ceil(workgroup_shape.z()),
    ]
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct SgemmParams {
    double_buffer: bool,
    block_m_size: u32,
    block_n_size: u32,
    block_k_size: u32,
    thread_m_size: u32,
    thread_n_size: u32,
}

impl SgemmParams {
    pub fn new(
        double_buffer: bool,
        block_m_size: u32,
        block_n_size: u32,
        block_k_size: u32,
        thread_m_size: u32,
        thread_n_size: u32,
    ) -> Self {
        Self {
            double_buffer,
            block_m_size,
            block_n_size,
            block_k_size,
            thread_m_size,
            thread_n_size,
        }
    }

    pub fn double_buffer(&self) -> bool {
        self.double_buffer
    }

    pub fn block_m_size(&self) -> u32 {
        self.block_m_size
    }

    pub fn block_n_size(&self) -> u32 {
        self.block_n_size
    }

    pub fn block_k_size(&self) -> u32 {
        self.block_k_size
    }

    pub fn thread_m_size(&self) -> u32 {
        self.thread_m_size
    }

    pub fn thread_n_size(&self) -> u32 {
        self.thread_n_size
    }
}

impl Default for SgemmParams {
    fn default() -> Self {
        let thread_m_size: u32 = 4;
        let thread_n_size: u32 = 4;
        let block_m_size: u32 = thread_m_size * 16;
        let block_n_size: u32 = thread_n_size * 8;
        let block_k_size: u32 = 8;
        let double_buffer: bool = false;

        Self {
            double_buffer,
            block_m_size,
            block_n_size,
            block_k_size,
            thread_m_size,
            thread_n_size,
        }
    }
}
