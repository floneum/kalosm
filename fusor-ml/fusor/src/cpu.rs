#[cfg(feature = "cpu")]
pub use fusor_cpu::*;

#[cfg(not(feature = "cpu"))]
mod disabled {
    use std::{convert::Infallible, fmt, marker::PhantomData, ops::Deref};

    pub use fusor_gguf::{
        BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType, GgufBlock,
    };
    pub use fusor_types::{FromArray, Layout, TensorSlice};

    pub type ABox<T> = Box<T>;

    #[derive(Clone, Debug)]
    pub struct AVec<T>(Vec<T>);

    impl<T> AVec<T> {
        pub fn with_capacity(_alignment: usize, capacity: usize) -> Self {
            Self(Vec::with_capacity(capacity))
        }

        pub fn extend_from_slice(&mut self, values: &[T])
        where
            T: Clone,
        {
            self.0.extend_from_slice(values);
        }

        pub fn into_boxed_slice(self) -> Box<[T]> {
            self.0.into_boxed_slice()
        }
    }

    pub trait SimdElement: Copy + Default + Send + Sync + 'static {}
    impl<T> SimdElement for T where T: Copy + Default + Send + Sync + 'static {}

    pub trait Scalar: Copy {}
    impl<T: Copy> Scalar for T {}

    pub trait FloatOps: SimdElement + PartialOrd {
        fn powf(self, exp: Self) -> Self;
        fn float_max(self, other: Self) -> Self;
        fn float_min(self, other: Self) -> Self;
    }

    impl FloatOps for f32 {
        fn powf(self, exp: Self) -> Self {
            f32::powf(self, exp)
        }

        fn float_max(self, other: Self) -> Self {
            f32::max(self, other)
        }

        fn float_min(self, other: Self) -> Self {
            f32::min(self, other)
        }
    }

    impl FloatOps for f64 {
        fn powf(self, exp: Self) -> Self {
            f64::powf(self, exp)
        }

        fn float_max(self, other: Self) -> Self {
            f64::max(self, other)
        }

        fn float_min(self, other: Self) -> Self {
            f64::min(self, other)
        }
    }

    impl FloatOps for half::f16 {
        fn powf(self, exp: Self) -> Self {
            half::f16::from_f32(self.to_f32().powf(exp.to_f32()))
        }

        fn float_max(self, other: Self) -> Self {
            if self >= other { self } else { other }
        }

        fn float_min(self, other: Self) -> Self {
            if self <= other { self } else { other }
        }
    }

    pub trait CastTo<T>: SimdElement {}
    impl<S, T> CastTo<T> for S where S: SimdElement {}

    pub trait IsNonZero: SimdElement {}
    impl<T: SimdElement> IsNonZero for T {}

    pub trait MatmulImpl: SimdElement {}
    impl<T: SimdElement> MatmulImpl for T {}

    pub trait SimdUnaryOp<E: SimdElement>: Copy {}
    pub trait SimdBinaryOp<E: SimdElement>: Copy {}
    pub trait SimdReduceOp<E: SimdElement>: Copy {}

    macro_rules! marker_ops {
        ($trait_name:ident: $($name:ident),* $(,)?) => {
            $(
                #[derive(Clone, Copy, Debug)]
                pub struct $name;

                impl<E: SimdElement> $trait_name<E> for $name {}
            )*
        };
    }

    marker_ops!(
        SimdUnaryOp:
        AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CosOp, CoshOp,
        Exp2Op, ExpOp, Log2Op, LogOp, NegOp, SinOp, SinhOp, SqrtOp, TanOp, TanhOp,
    );

    marker_ops!(
        SimdBinaryOp:
        AddOp, DivOp, EqOp, GtOp, GteOp, LtOp, LteOp, MulOp, NeOp, RemOp, SubOp,
    );

    marker_ops!(SimdReduceOp: MaxOp, MinOp, ProdOp, SumOp);

    #[derive(Clone, Copy, Debug)]
    pub enum ConcreteTensor<T, const R: usize> {
        Disabled(PhantomData<T>, Infallible),
    }

    macro_rules! never {
        ($value:expr) => {
            match $value {
                Self::Disabled(_, never) => match *never {},
            }
        };
    }

    impl<T, const R: usize> ConcreteTensor<T, R> {
        pub fn zeros(_shape: [usize; R]) -> Self {
            unreachable!("CPU backend is disabled")
        }

        pub fn from_slice(_shape: [usize; R], _data: &[T]) -> Self {
            unreachable!("CPU backend is disabled")
        }

        pub fn from_parts(_layout: Layout, _data: ABox<[T]>) -> Self {
            unreachable!("CPU backend is disabled")
        }

        pub fn data(&self) -> &ABox<[T]> {
            never!(self)
        }

        pub fn data_mut(&mut self) -> &mut ABox<[T]> {
            never!(self)
        }

        pub fn q_mat_mul<B: GgufBlock>(&self, _rhs: &QuantizedTensor<B>) -> ConcreteTensor<f32, R> {
            never!(self)
        }
    }

    pub trait LazyBacking: Sync {
        type Elem;
    }

    pub trait TensorBacking<const R: usize>: LazyBacking {
        fn layout(&self) -> Layout;
        fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R>;
    }

    pub trait ResolvedTensor<const R: usize>: TensorBacking<R> {
        fn data(&self) -> &ABox<[Self::Elem]>;
        fn data_mut(&mut self) -> &mut ABox<[Self::Elem]>;
    }

    impl<T: SimdElement, const R: usize> LazyBacking for ConcreteTensor<T, R> {
        type Elem = T;
    }

    impl<T: SimdElement, const R: usize> TensorBacking<R> for ConcreteTensor<T, R> {
        fn layout(&self) -> Layout {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }

        fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R> {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }
    }

    impl<T: SimdElement, const R: usize> ResolvedTensor<R> for ConcreteTensor<T, R> {
        fn data(&self) -> &ABox<[Self::Elem]> {
            self.data()
        }

        fn data_mut(&mut self) -> &mut ABox<[Self::Elem]> {
            self.data_mut()
        }
    }

    impl<T> LazyBacking for &T
    where
        T: LazyBacking + Sync,
    {
        type Elem = T::Elem;
    }

    impl<const R: usize, T> TensorBacking<R> for &T
    where
        T: TensorBacking<R> + Sync,
    {
        fn layout(&self) -> Layout {
            (*self).layout()
        }

        fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R> {
            (*self).to_concrete()
        }
    }

    macro_rules! backing_type {
        ($($name:ident),* $(,)?) => {
            $(
                #[derive(Clone, Copy, Debug)]
                pub struct $name<E, const R: usize, T = (), U = ()> {
                    _marker: PhantomData<(E, T, U)>,
                }

                impl<E: SimdElement, const R: usize, T: Sync, U: Sync> LazyBacking for $name<E, R, T, U> {
                    type Elem = E;
                }

                impl<E: SimdElement, const R: usize, T: Sync, U: Sync> TensorBacking<R>
                    for $name<E, R, T, U>
                {
                    fn layout(&self) -> Layout {
                        unreachable!("CPU backend is disabled")
                    }

                    fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R> {
                        unreachable!("CPU backend is disabled")
                    }
                }
            )*
        };
    }

    #[derive(Clone, Copy, Debug)]
    pub struct MapLayout<T, const R: usize> {
        _marker: PhantomData<T>,
    }

    impl<T: LazyBacking, const R: usize> LazyBacking for MapLayout<T, R> {
        type Elem = T::Elem;
    }

    impl<T: LazyBacking, const R: usize> TensorBacking<R> for MapLayout<T, R> {
        fn layout(&self) -> Layout {
            unreachable!("CPU backend is disabled")
        }

        fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R> {
            unreachable!("CPU backend is disabled")
        }
    }

    backing_type!(
        Abs, Acos, Acosh, Add, AddScalar, Asin, Asinh, Atan, Atanh, Broadcast, Cos, Cosh, Div,
        DivScalar, Exp, Exp2, Log, Log2, Mul, MulScalar, Neg, Rem, Sin, Sinh, Sqrt, Sub, SubScalar,
        Tan, Tanh,
    );

    pub type Eq<E, const R: usize, T, U> = Add<E, R, T, U>;
    pub type Gt<E, const R: usize, T, U> = Add<E, R, T, U>;
    pub type Gte<E, const R: usize, T, U> = Add<E, R, T, U>;
    pub type Lt<E, const R: usize, T, U> = Add<E, R, T, U>;
    pub type Lte<E, const R: usize, T, U> = Add<E, R, T, U>;
    pub type Ne<E, const R: usize, T, U> = Add<E, R, T, U>;

    #[derive(Clone, Copy, Debug)]
    pub enum Tensor<const R: usize, T: TensorBacking<R>> {
        Disabled(PhantomData<T>, Infallible),
    }

    impl<const R: usize, T: TensorBacking<R>> Tensor<R, T> {
        pub fn new(_inner: T) -> Self {
            unreachable!("CPU backend is disabled")
        }

        pub fn inner(&self) -> &T {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }

        pub fn inner_mut(&mut self) -> &mut T {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }

        pub fn into_inner(self) -> T {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn as_ref(&self) -> Tensor<R, &T> {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }
    }

    impl<const R: usize, T> LazyBacking for Tensor<R, T>
    where
        T: TensorBacking<R>,
    {
        type Elem = T::Elem;
    }

    impl<const R: usize, T> TensorBacking<R> for Tensor<R, T>
    where
        T: TensorBacking<R>,
    {
        fn layout(&self) -> Layout {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }

        fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R> {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }
    }

    impl<const R: usize, D, T> FromArray<R, D, T, ()> for Tensor<R, ConcreteTensor<D, R>>
    where
        D: SimdElement,
        T: fusor_types::IntoFlatArray<D, R>,
    {
        fn from_array(_data: T, _device: &()) -> Self {
            unreachable!("CPU backend is disabled")
        }
    }

    impl<const R: usize, E: SimdElement> Tensor<R, ConcreteTensor<E, R>> {
        pub fn zeros(_shape: [usize; R]) -> Self {
            unreachable!("CPU backend is disabled")
        }

        pub fn from_slice(_shape: [usize; R], _data: &[E]) -> Self {
            unreachable!("CPU backend is disabled")
        }
    }

    impl<const R: usize, E, T> Tensor<R, T>
    where
        E: SimdElement,
        T: TensorBacking<R, Elem = E>,
    {
        pub fn shape(&self) -> [usize; R] {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }

        pub fn to_concrete(&self) -> Tensor<R, ConcreteTensor<E, R>> {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }

        pub fn restride<const R2: usize>(
            self,
            _specs: [fusor_types::StrideSpec; R2],
        ) -> Tensor<R2, MapLayout<T, R2>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn restride_layout<const R2: usize>(
            self,
            _new_layout: Layout,
        ) -> Tensor<R2, MapLayout<T, R2>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn reshape<const R2: usize>(
            self,
            _new_shape: [usize; R2],
        ) -> Tensor<R2, MapLayout<T, R2>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn broadcast_as<const R2: usize>(
            self,
            _target_shape: [usize; R2],
        ) -> Tensor<R2, Broadcast<E, R2, T>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn make_contiguous(self) -> Tensor<R, ConcreteTensor<E, R>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn flatten_all(&self) -> Tensor<1, ConcreteTensor<E, 1>> {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }

        pub fn as_slice(&self) -> TensorSlice<R, E, CpuMappedBuffer> {
            match self {
                Self::Disabled(_, never) => match *never {},
            }
        }

        pub fn index_select(
            self,
            _dimension: usize,
            _indexes: Tensor<1, impl TensorBacking<1, Elem = u32>>,
        ) -> Tensor<R, ConcreteTensor<E, R>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn slice_assign<T2>(
            self,
            _slices: [std::ops::Range<usize>; R],
            _value: Tensor<R, T2>,
        ) -> Tensor<R, ConcreteTensor<E, R>>
        where
            T2: TensorBacking<R, Elem = E>,
        {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn where_cond<T2, T3>(
            self,
            _on_true: Tensor<R, T2>,
            _on_false: Tensor<R, T3>,
        ) -> Tensor<R, ConcreteTensor<E, R>>
        where
            T2: TensorBacking<R, Elem = E>,
            T3: TensorBacking<R, Elem = E>,
        {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn cast<E2>(self) -> Tensor<R, ConcreteTensor<E2, R>>
        where
            E2: SimdElement,
        {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn max_axis<const R2: usize>(self, _axis: usize) -> Tensor<R2, ConcreteTensor<E, R2>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn min_axis<const R2: usize>(self, _axis: usize) -> Tensor<R2, ConcreteTensor<E, R2>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn sum_axis<const R2: usize>(self, _axis: usize) -> Tensor<R2, ConcreteTensor<E, R2>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn prod_axis<const R2: usize>(self, _axis: usize) -> Tensor<R2, ConcreteTensor<E, R2>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn matmul<T2>(self, _rhs: Tensor<R, T2>) -> Tensor<R, ConcreteTensor<E, R>>
        where
            T2: TensorBacking<R, Elem = E>,
        {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn pow_scalar(self, _exponent: E) -> Tensor<R, ConcreteTensor<E, R>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn max_scalar(self, _scalar: E) -> Tensor<R, ConcreteTensor<E, R>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn min_scalar(self, _scalar: E) -> Tensor<R, ConcreteTensor<E, R>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn clamp(self, _min: E, _max: E) -> Tensor<R, ConcreteTensor<E, R>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn add_scalar(self, _scalar: E) -> Tensor<R, AddScalar<E, R, T>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn sub_scalar(self, _scalar: E) -> Tensor<R, SubScalar<E, R, T>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn mul_scalar(self, _scalar: E) -> Tensor<R, MulScalar<E, R, T>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn div_scalar(self, _scalar: E) -> Tensor<R, DivScalar<E, R, T>> {
            match self {
                Self::Disabled(_, never) => match never {},
            }
        }

        pub fn eq<T2>(self, _rhs: Tensor<R, T2>) -> Tensor<R, Eq<E, R, T, T2>>
        where
            T2: TensorBacking<R, Elem = E>,
        {
            never_tensor(self)
        }

        pub fn ne<T2>(self, _rhs: Tensor<R, T2>) -> Tensor<R, Ne<E, R, T, T2>>
        where
            T2: TensorBacking<R, Elem = E>,
        {
            never_tensor(self)
        }

        pub fn lt<T2>(self, _rhs: Tensor<R, T2>) -> Tensor<R, Lt<E, R, T, T2>>
        where
            T2: TensorBacking<R, Elem = E>,
        {
            never_tensor(self)
        }

        pub fn lte<T2>(self, _rhs: Tensor<R, T2>) -> Tensor<R, Lte<E, R, T, T2>>
        where
            T2: TensorBacking<R, Elem = E>,
        {
            never_tensor(self)
        }

        pub fn gt<T2>(self, _rhs: Tensor<R, T2>) -> Tensor<R, Gt<E, R, T, T2>>
        where
            T2: TensorBacking<R, Elem = E>,
        {
            never_tensor(self)
        }

        pub fn gte<T2>(self, _rhs: Tensor<R, T2>) -> Tensor<R, Gte<E, R, T, T2>>
        where
            T2: TensorBacking<R, Elem = E>,
        {
            never_tensor(self)
        }

        pub fn eq_scalar(self, _scalar: E) -> Tensor<R, Eq<E, R, T, ()>> {
            never_tensor(self)
        }

        pub fn ne_scalar(self, _scalar: E) -> Tensor<R, Ne<E, R, T, ()>> {
            never_tensor(self)
        }

        pub fn lt_scalar(self, _scalar: E) -> Tensor<R, Lt<E, R, T, ()>> {
            never_tensor(self)
        }

        pub fn lte_scalar(self, _scalar: E) -> Tensor<R, Lte<E, R, T, ()>> {
            never_tensor(self)
        }

        pub fn gt_scalar(self, _scalar: E) -> Tensor<R, Gt<E, R, T, ()>> {
            never_tensor(self)
        }

        pub fn gte_scalar(self, _scalar: E) -> Tensor<R, Gte<E, R, T, ()>> {
            never_tensor(self)
        }

        pub fn abs(self) -> Tensor<R, Abs<E, R, T>> {
            never_tensor(self)
        }
        pub fn sqrt(self) -> Tensor<R, Sqrt<E, R, T>> {
            never_tensor(self)
        }
        pub fn exp(self) -> Tensor<R, Exp<E, R, T>> {
            never_tensor(self)
        }
        pub fn exp2(self) -> Tensor<R, Exp2<E, R, T>> {
            never_tensor(self)
        }
        pub fn log(self) -> Tensor<R, Log<E, R, T>> {
            never_tensor(self)
        }
        pub fn log2(self) -> Tensor<R, Log2<E, R, T>> {
            never_tensor(self)
        }
        pub fn sin(self) -> Tensor<R, Sin<E, R, T>> {
            never_tensor(self)
        }
        pub fn cos(self) -> Tensor<R, Cos<E, R, T>> {
            never_tensor(self)
        }
        pub fn tan(self) -> Tensor<R, Tan<E, R, T>> {
            never_tensor(self)
        }
        pub fn tanh(self) -> Tensor<R, Tanh<E, R, T>> {
            never_tensor(self)
        }
        pub fn asin(self) -> Tensor<R, Asin<E, R, T>> {
            never_tensor(self)
        }
        pub fn acos(self) -> Tensor<R, Acos<E, R, T>> {
            never_tensor(self)
        }
        pub fn atan(self) -> Tensor<R, Atan<E, R, T>> {
            never_tensor(self)
        }
        pub fn sinh(self) -> Tensor<R, Sinh<E, R, T>> {
            never_tensor(self)
        }
        pub fn cosh(self) -> Tensor<R, Cosh<E, R, T>> {
            never_tensor(self)
        }
        pub fn asinh(self) -> Tensor<R, Asinh<E, R, T>> {
            never_tensor(self)
        }
        pub fn acosh(self) -> Tensor<R, Acosh<E, R, T>> {
            never_tensor(self)
        }
        pub fn atanh(self) -> Tensor<R, Atanh<E, R, T>> {
            never_tensor(self)
        }
    }

    fn never_tensor<const R: usize, T, const R2: usize, U>(tensor: Tensor<R, T>) -> Tensor<R2, U>
    where
        T: TensorBacking<R>,
        U: TensorBacking<R2>,
    {
        match tensor {
            Tensor::Disabled(_, never) => match never {},
        }
    }

    macro_rules! tensor_binary_op {
        ($trait_name:ident, $method:ident, $output:ident) => {
            impl<const R: usize, E, T1, T2> std::ops::$trait_name<Tensor<R, T2>> for Tensor<R, T1>
            where
                E: SimdElement,
                T1: TensorBacking<R, Elem = E>,
                T2: TensorBacking<R, Elem = E>,
            {
                type Output = Tensor<R, $output<E, R, T1, T2>>;

                fn $method(self, _rhs: Tensor<R, T2>) -> Self::Output {
                    never_tensor(self)
                }
            }

            impl<'a, const R: usize, E, T1, T2> std::ops::$trait_name<&'a Tensor<R, T2>>
                for Tensor<R, T1>
            where
                E: SimdElement,
                T1: TensorBacking<R, Elem = E>,
                T2: TensorBacking<R, Elem = E>,
            {
                type Output = Tensor<R, $output<E, R, T1, &'a T2>>;

                fn $method(self, _rhs: &'a Tensor<R, T2>) -> Self::Output {
                    never_tensor(self)
                }
            }

            impl<'a, const R: usize, E, T1, T2> std::ops::$trait_name<Tensor<R, T2>>
                for &'a Tensor<R, T1>
            where
                E: SimdElement,
                T1: TensorBacking<R, Elem = E>,
                T2: TensorBacking<R, Elem = E>,
            {
                type Output = Tensor<R, $output<E, R, &'a T1, T2>>;

                fn $method(self, _rhs: Tensor<R, T2>) -> Self::Output {
                    match self {
                        Tensor::Disabled(_, never) => match *never {},
                    }
                }
            }

            impl<'a, const R: usize, E, T1, T2> std::ops::$trait_name<&'a Tensor<R, T2>>
                for &'a Tensor<R, T1>
            where
                E: SimdElement,
                T1: TensorBacking<R, Elem = E>,
                T2: TensorBacking<R, Elem = E>,
            {
                type Output = Tensor<R, $output<E, R, &'a T1, &'a T2>>;

                fn $method(self, _rhs: &'a Tensor<R, T2>) -> Self::Output {
                    match self {
                        Tensor::Disabled(_, never) => match *never {},
                    }
                }
            }
        };
    }

    tensor_binary_op!(Add, add, Add);
    tensor_binary_op!(Sub, sub, Sub);
    tensor_binary_op!(Mul, mul, Mul);
    tensor_binary_op!(Div, div, Div);
    tensor_binary_op!(Rem, rem, Rem);

    macro_rules! tensor_scalar_op {
        ($trait_name:ident, $method:ident, $output:ident) => {
            impl<const R: usize, E, T> std::ops::$trait_name<E> for Tensor<R, T>
            where
                E: SimdElement,
                T: TensorBacking<R, Elem = E>,
            {
                type Output = Tensor<R, $output<E, R, T>>;

                fn $method(self, _rhs: E) -> Self::Output {
                    never_tensor(self)
                }
            }

            impl<'a, const R: usize, E, T> std::ops::$trait_name<E> for &'a Tensor<R, T>
            where
                E: SimdElement,
                T: TensorBacking<R, Elem = E>,
            {
                type Output = Tensor<R, $output<E, R, &'a T>>;

                fn $method(self, _rhs: E) -> Self::Output {
                    match self {
                        Tensor::Disabled(_, never) => match *never {},
                    }
                }
            }
        };
    }

    tensor_scalar_op!(Add, add, AddScalar);
    tensor_scalar_op!(Sub, sub, SubScalar);
    tensor_scalar_op!(Mul, mul, MulScalar);
    tensor_scalar_op!(Div, div, DivScalar);

    impl<const R: usize, E, T> std::ops::Neg for Tensor<R, T>
    where
        E: SimdElement,
        T: TensorBacking<R, Elem = E>,
    {
        type Output = Tensor<R, Neg<E, R, T>>;

        fn neg(self) -> Self::Output {
            never_tensor(self)
        }
    }

    impl<'a, const R: usize, E, T> std::ops::Neg for &'a Tensor<R, T>
    where
        E: SimdElement,
        T: TensorBacking<R, Elem = E>,
    {
        type Output = Tensor<R, Neg<E, R, &'a T>>;

        fn neg(self) -> Self::Output {
            match self {
                Tensor::Disabled(_, never) => match *never {},
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct QuantizedTensor<B> {
        shape: Box<[usize]>,
        _marker: PhantomData<B>,
    }

    impl<B> QuantizedTensor<B> {
        pub fn from_raw_bytes(_element_shape: impl Into<Box<[usize]>>, _bytes: &[u8]) -> Self {
            unreachable!("CPU backend is disabled")
        }

        pub fn element_shape(&self) -> &[usize] {
            &self.shape
        }

        pub fn dequantize<const R: usize>(&self) -> ConcreteTensor<f32, R> {
            unreachable!("CPU backend is disabled")
        }
    }

    #[derive(Clone)]
    pub struct CpuMappedBuffer {
        bytes: Box<[u8]>,
    }

    impl CpuMappedBuffer {
        pub fn new(bytes: Box<[u8]>) -> Self {
            Self { bytes }
        }
    }

    impl Deref for CpuMappedBuffer {
        type Target = [u8];

        fn deref(&self) -> &Self::Target {
            &self.bytes
        }
    }

    pub fn materialize_expr<const R: usize, T>(_expr: &T) -> ConcreteTensor<T::Elem, R>
    where
        T: TensorBacking<R>,
    {
        unreachable!("CPU backend is disabled")
    }

    pub fn layer_norm_last_dim_fused<const R: usize, E, T, W, B>(
        _input: &T,
        _weight: &W,
        _bias: Option<&B>,
        _eps: f32,
    ) -> ConcreteTensor<E, R>
    where
        E: SimdElement,
        T: TensorBacking<R, Elem = E>,
    {
        unreachable!("CPU backend is disabled")
    }

    pub fn softmax_last_dim_fused<T, const R: usize>(_input: &T) -> ConcreteTensor<T::Elem, R>
    where
        T: TensorBacking<R>,
    {
        unreachable!("CPU backend is disabled")
    }

    pub trait LastRankInner {
        type LastRank;
    }

    impl<T: SimdElement, const R: usize> LastRankInner for ConcreteTensor<T, R> {
        type LastRank = ConcreteTensor<T, R>;
    }

    pub trait LastRank<const R: usize, T: SimdElement> {}
    impl<const IN: usize, const OUT: usize, T: SimdElement> LastRank<OUT, T> for ConcreteTensor<T, IN> {}

    pub trait NextRankInner {
        type NextRank;
    }

    impl<T: SimdElement, const R: usize> NextRankInner for ConcreteTensor<T, R> {
        type NextRank = ConcreteTensor<T, R>;
    }

    pub trait NextRank<const R: usize, T: SimdElement> {}
    impl<const IN: usize, const OUT: usize, T: SimdElement> NextRank<OUT, T> for ConcreteTensor<T, IN> {}

    pub trait SmallerRankInner<const DIFF: usize> {
        type SmallerRank;
    }

    impl<T: SimdElement, const R: usize, const DIFF: usize> SmallerRankInner<DIFF>
        for ConcreteTensor<T, R>
    {
        type SmallerRank = ConcreteTensor<T, R>;
    }

    pub trait SmallerRank<const R: usize, const DIFF: usize, T: SimdElement> {}
    impl<const IN: usize, const OUT: usize, const DIFF: usize, T: SimdElement>
        SmallerRank<OUT, DIFF, T> for ConcreteTensor<T, IN>
    {
    }

    pub trait LargerRankInner<const DIFF: usize> {
        type LargerRank;
    }

    impl<T: SimdElement, const R: usize, const DIFF: usize> LargerRankInner<DIFF>
        for ConcreteTensor<T, R>
    {
        type LargerRank = ConcreteTensor<T, R>;
    }

    pub trait LargerRank<const R: usize, const DIFF: usize, T: SimdElement> {}
    impl<const IN: usize, const OUT: usize, const DIFF: usize, T: SimdElement>
        LargerRank<OUT, DIFF, T> for ConcreteTensor<T, IN>
    {
    }

    pub trait MaxRankInner {
        type MaxRank;
    }

    impl<T: SimdElement, const R: usize, const R2: usize> MaxRankInner
        for (ConcreteTensor<T, R>, ConcreteTensor<T, R2>)
    {
        type MaxRank = ConcreteTensor<T, R>;
    }

    pub trait MaxRank<const R: usize, T: SimdElement> {}
    impl<const A: usize, const B: usize, const OUT: usize, T: SimdElement> MaxRank<OUT, T>
        for (ConcreteTensor<T, A>, ConcreteTensor<T, B>)
    {
    }

    impl fmt::Debug for CpuMappedBuffer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("CpuMappedBuffer").finish_non_exhaustive()
        }
    }
}

#[cfg(not(feature = "cpu"))]
pub use disabled::*;
