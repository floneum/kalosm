use std::pin::Pin;

use fusor::{DataType, Device, SimdElement, Tensor, WasmNotSend};

/// A future that is `Send` on non-wasm32 targets, but is not required to be on wasm32.
#[doc(hidden)]
pub trait FutureWasmNotSend: std::future::Future + WasmNotSend {}
impl<T: std::future::Future + WasmNotSend> FutureWasmNotSend for T {}

pub(crate) type BoxFuture<'a, T> = Pin<Box<dyn FutureWasmNotSend<Output = T> + 'a>>;

#[doc(hidden)]
pub trait AsyncFnMutTuple<Args> {
    type Output;

    fn call_mut<'a>(&'a mut self, args: Args) -> BoxFuture<'a, Self::Output>;
}

macro_rules! impl_fn_mut_tuple {
    ($($type:ident),*) => {
        impl<Fn, U, Fut, $($type),*> AsyncFnMutTuple<($($type,)*)> for Fn
        where
            Fn: FnMut($($type,)*) -> Fut,
            Fut: std::future::Future<Output = U> + WasmNotSend + 'static,
        {
            type Output = U;
            #[allow(non_snake_case)]
            fn call_mut<'a>(&'a mut self, ($($type,)*): ($($type,)*)) -> BoxFuture<'a, Self::Output> {
                Box::pin((self)($($type,)*))
            }
        }
    };
}

impl_fn_mut_tuple!();
impl_fn_mut_tuple!(A);
impl_fn_mut_tuple!(A, B);
impl_fn_mut_tuple!(A, B, C);
impl_fn_mut_tuple!(A, B, C, D);
impl_fn_mut_tuple!(A, B, C, D, E);
impl_fn_mut_tuple!(A, B, C, D, E, F);

#[doc(hidden)]
pub trait GenTuple {
    type Output;
    fn generate(&mut self, device: &Device, run: usize) -> Self::Output;
    fn run_label(&self, run: usize) -> String;
}

macro_rules! impl_gen_tuple {
    ($($type:ident -> $type_out:ident),*) => {
        impl<$($type, $type_out),*> GenTuple for ($($type,)*)
        where
            $(
                $type: crate::GenerateFromDevice<Output = $type_out>,
            )*
        {
            type Output = ($($type_out,)*);
            #[allow(non_snake_case)]
            fn generate(&mut self, device: &Device, run: usize) -> Self::Output {
                let ($($type,)*) = self;
                (
                    $(
                        $type.generate(device, run),
                    )*
                )
            }

            #[allow(non_snake_case)]
            fn run_label(&self, run: usize) -> String {
                let ($($type,)*) = self;
                let mut fragments = Vec::new();
                $(
                    if let Some(fragment) = $type.run_label_fragment(run) {
                        if !fragments.contains(&fragment) {
                            fragments.push(fragment);
                        }
                    }
                )*

                if fragments.is_empty() {
                    format!("run{run}")
                } else {
                    format!("sample{run}_{}", fragments.join("_"))
                }
            }
        }
    };
}

impl_gen_tuple!(A -> AOut);
impl_gen_tuple!(A -> AOut, B -> BOut);
impl_gen_tuple!(A -> AOut, B -> BOut, C -> COut);
impl_gen_tuple!(A -> AOut, B -> BOut, C -> COut, D -> DOut);
impl_gen_tuple!(A -> AOut, B -> BOut, C -> COut, D -> DOut, E -> EOut);
impl_gen_tuple!(A -> AOut, B -> BOut, C -> COut, D -> DOut, E -> EOut, F -> FOut);

#[doc(hidden)]
pub trait ResolveTensorTuple {
    type Output;

    fn resolve(self) -> BoxFuture<'static, Result<Self::Output, fusor::Error>>;

    fn extract_device(&self) -> Device;
}

macro_rules! impl_resolve_tensor_tuple {
    ($($type:ident = $rank:ident),*) => {
        impl<$(const $rank: usize,)* $($type: DataType + SimdElement),*> ResolveTensorTuple for ($(Tensor<$rank, $type>,)*)
            where
                $(
                    fusor_types::TensorSlice<$rank, $type, fusor::EitherMappedBuffer>: fusor::ToVec<Output: Send + 'static>,
                )*
        {
            type Output = ($(<fusor_types::TensorSlice<$rank, $type, fusor::EitherMappedBuffer> as fusor::ToVec>::Output,)*);

            #[allow(non_snake_case)]
            fn resolve(self) -> BoxFuture<'static, Result<Self::Output, fusor::Error>> {
                Box::pin(async move {
                let ($($type,)*) = self;
                Ok((
                    $(
                        fusor::ToVec::to_vec(&$type.as_slice().await?),
                    )*
                ))
                })
            }

            fn extract_device(&self) -> Device {
                self.0.device()
            }
        }
    };
}

impl_resolve_tensor_tuple!(A = N1);
impl_resolve_tensor_tuple!(A = N1, B = N2);
impl_resolve_tensor_tuple!(A = N1, B = N2, C = N3);
impl_resolve_tensor_tuple!(A = N1, B = N2, C = N3, D = N4);
impl_resolve_tensor_tuple!(A = N1, B = N2, C = N3, D = N4, E = N5);
impl_resolve_tensor_tuple!(A = N1, B = N2, C = N3, D = N4, E = N5, F = N6);

#[doc(hidden)]
pub trait PushTuple<Tail> {
    type Output;
    fn push(self, new_last: Tail) -> Self::Output;
}

#[doc(hidden)]
pub trait PopTuple {
    type First;
    type Rest;
    fn pop(self) -> (Self::First, Self::Rest);
}

macro_rules! impl_push_pop_tuple {
    ($first_type:ident $(,$type:ident)*) => {
        impl<$first_type $(,$type)*> PopTuple for ($first_type, $($type,)*) {
            type First = $first_type;
            type Rest = ($($type,)*);
            #[allow(non_snake_case)]
            fn pop(self) -> (Self::First, Self::Rest) {
                let ($first_type, $($type,)*) = self;
                ($first_type, ($($type,)*))
            }
        }

        impl<$first_type, $($type,)* Tail> PushTuple<Tail> for ($first_type, $($type,)*) {
            type Output = ($first_type, $($type,)* Tail);
            #[allow(non_snake_case)]
            fn push(self, new_last: Tail) -> Self::Output {
                let (head, $($type,)*) = self;
                (head, $($type,)* new_last)
            }
        }
    };
}

impl PopTuple for () {
    type First = ();
    type Rest = ();
    fn pop(self) -> (Self::First, Self::Rest) {
        ((), ())
    }
}
impl<Tail> PushTuple<Tail> for () {
    type Output = (Tail,);
    fn push(self, new_last: Tail) -> Self::Output {
        (new_last,)
    }
}
impl_push_pop_tuple!(A);
impl_push_pop_tuple!(A, B);
impl_push_pop_tuple!(A, B, C);
impl_push_pop_tuple!(A, B, C, D);
impl_push_pop_tuple!(A, B, C, D, E);
impl_push_pop_tuple!(A, B, C, D, E, F);
