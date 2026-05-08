use wgpu::naga::{Arena, BinaryOperator, Block, Expression, GlobalVariable, Handle};

use super::SampleMirostat2Globals;
use crate::mir::kernel_backend::naga_helpers::NagaBuilderExt;

impl super::SampleMirostat2ModuleBuilder {
    pub(super) fn top_weight(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        max_value: Handle<Expression>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let value = self.top_value(expressions, body, globals, index);
        let delta = self.bin(
            expressions,
            body,
            BinaryOperator::Subtract,
            value,
            max_value,
        );
        self.exp_f32(expressions, body, delta)
    }

    pub(super) fn top_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let index = self.index1(
            expressions,
            body,
            self.meta.values_offset,
            self.meta.values_stride,
            index,
        );
        self.load_storage(expressions, body, globals.values, index)
    }

    pub(super) fn top_id(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let index = self.index1(
            expressions,
            body,
            self.meta.ids_offset,
            self.meta.ids_stride,
            index,
        );
        self.load_storage(expressions, body, globals.ids, index)
    }

    pub(super) fn store_sample_result(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        output: Handle<GlobalVariable>,
        status: u32,
        token: u32,
    ) {
        let token = self.u32_lit(expressions, token);
        self.store_sample_result_handle(expressions, body, output, status, token);
    }

    pub(super) fn store_sample_result_handle(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        output: Handle<GlobalVariable>,
        status: u32,
        token: Handle<Expression>,
    ) {
        let zero = self.u32_lit(expressions, 0);
        let one = self.u32_lit(expressions, 1);
        let status = self.u32_lit(expressions, status);
        self.store_storage(expressions, body, output, zero, status);
        self.store_storage(expressions, body, output, one, token);
    }

    pub(super) fn index1(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        stride: u32,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let scaled = if stride == 1 {
            index
        } else {
            self.mul_lit(expressions, body, index, stride)
        };
        self.add_lit(expressions, body, scaled, offset)
    }

    pub(super) fn load_param_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: u32,
    ) -> Handle<Expression> {
        let index = self.u32_lit(expressions, index);
        self.load_storage(expressions, body, global, index)
    }
}
