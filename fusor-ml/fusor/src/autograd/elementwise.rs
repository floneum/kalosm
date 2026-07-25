use super::*;

impl<const R: usize, T: AutogradElement> Tensor<R, T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
{
    pub fn add(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() + rhs.value.clone()).into_concrete(),
            |grad, _, _| vec![grad.clone().into_concrete(), grad.into_concrete()],
        )
    }

    pub fn add_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.add(&rhs)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() - rhs.value.clone()).into_concrete(),
            |grad, _, _| vec![grad.clone().into_concrete(), (-grad).into_concrete()],
        )
    }

    pub fn sub_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.sub(&rhs)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() * rhs.value.clone()).into_concrete(),
            |grad, lhs, rhs| {
                vec![
                    (grad.clone() * rhs).into_concrete(),
                    (grad * lhs).into_concrete(),
                ]
            },
        )
    }

    pub fn mul_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.mul(&rhs)
    }

    pub fn div(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() / rhs.value.clone()).into_concrete(),
            |grad, lhs, rhs| {
                let lhs_grad = (grad.clone() / rhs.clone()).into_concrete();
                let rhs_grad = (-((grad * lhs) / rhs.sqr().into_concrete())).into_concrete();
                vec![lhs_grad, rhs_grad]
            },
        )
    }

    pub fn div_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.div(&rhs)
    }

    pub fn pow(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            self.value.pow(&rhs.value).into_concrete(),
            |grad, lhs, rhs| {
                let rhs_minus_one = rhs.sub_scalar(T::from_f32(1.0)).into_concrete();
                let lhs_power = lhs.pow(&rhs_minus_one).into_concrete();
                let lhs_grad =
                    ((grad.clone() * rhs.clone()).into_concrete() * lhs_power).into_concrete();
                let rhs_grad = ((grad * lhs.pow(&rhs).into_concrete()).into_concrete()
                    * lhs.log().into_concrete())
                .into_concrete();
                vec![lhs_grad, rhs_grad]
            },
        )
    }

    pub fn pow_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, T>) -> Tensor<R3, T> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.pow(&rhs)
    }

    pub fn pow_elementwise(&self, exponent: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(
            self.value
                .pow_elementwise(T::from_f32(exponent))
                .into_concrete(),
            move |grad, _| {
                let power = input
                    .pow_elementwise(T::from_f32(exponent - 1.0))
                    .into_concrete();
                (grad * power)
                    .into_concrete()
                    .mul_scalar(T::from_f32(exponent))
                    .into_concrete()
            },
        )
    }

    pub fn pow_scalar(&self, exponent: f32) -> Self {
        self.pow_elementwise(exponent)
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.add_scalar(T::from_f32(scalar)),
            move |grad, _| grad,
        )
    }

    pub fn sub_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.sub_scalar(T::from_f32(scalar)),
            move |grad, _| grad,
        )
    }

    pub fn mul_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.mul_scalar(T::from_f32(scalar)).into_concrete(),
            move |grad, _| grad.mul_scalar(T::from_f32(scalar)).into_concrete(),
        )
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.div_scalar(T::from_f32(scalar)).into_concrete(),
            move |grad, _| grad.div_scalar(T::from_f32(scalar)).into_concrete(),
        )
    }

    pub fn neg(&self) -> Self {
        self.unary_from_value((-self.value.clone()).into_concrete(), move |grad, _| {
            (-grad).into_concrete()
        })
    }

    pub fn sqr(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sqr().into_concrete(), move |grad, _| {
            ((grad * input.clone())
                .into_concrete()
                .mul_scalar(T::from_f32(2.0)))
            .into_concrete()
        })
    }

    pub fn abs(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.abs().into_concrete(), move |grad, _| {
            let positive = input.mt(T::from_f32(0.0)).into_concrete();
            let negative = input.lt(T::from_f32(0.0)).into_concrete();
            ((grad.clone() * positive).into_concrete() - (grad * negative).into_concrete())
                .into_concrete()
        })
    }

    pub fn acos(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.acos().into_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), T::from_f32(1.0), input.shape())
                - input.sqr().into_concrete())
            .into_concrete()
            .sqrt()
            .into_concrete();
            (-(grad / denom).into_concrete()).into_concrete()
        })
    }

    pub fn acosh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.acosh().into_concrete(), move |grad, _| {
            let lower = input
                .add_scalar(T::from_f32(-1.0))
                .into_concrete()
                .sqrt()
                .into_concrete();
            let upper = input
                .add_scalar(T::from_f32(1.0))
                .into_concrete()
                .sqrt()
                .into_concrete();
            (grad / (lower * upper).into_concrete()).into_concrete()
        })
    }

    pub fn approximate_exp(&self) -> Self {
        self.unary_from_value(
            self.value.approximate_exp().into_concrete(),
            move |grad, out| (grad * out).into_concrete(),
        )
    }

    pub fn asin(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.asin().into_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), T::from_f32(1.0), input.shape())
                - input.sqr().into_concrete())
            .into_concrete()
            .sqrt()
            .into_concrete();
            (grad / denom).into_concrete()
        })
    }

    pub fn asinh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.asinh().into_concrete(), move |grad, _| {
            let denom = input
                .sqr()
                .add_scalar(T::from_f32(1.0))
                .into_concrete()
                .sqrt()
                .into_concrete();
            (grad / denom).into_concrete()
        })
    }

    pub fn atan(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.atan().into_concrete(), move |grad, _| {
            let denom = input.sqr().add_scalar(T::from_f32(1.0)).into_concrete();
            (grad / denom).into_concrete()
        })
    }

    pub fn atanh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.atanh().into_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), T::from_f32(1.0), input.shape())
                - input.sqr().into_concrete())
            .into_concrete();
            (grad / denom).into_concrete()
        })
    }

    pub fn cos(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.cos().into_concrete(), move |grad, _| {
            (-(grad * input.sin().into_concrete()).into_concrete()).into_concrete()
        })
    }

    pub fn cosh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.cosh().into_concrete(), move |grad, _| {
            (grad * input.sinh().into_concrete()).into_concrete()
        })
    }

    pub fn exp2(&self) -> Self {
        self.unary_from_value(self.value.exp2().into_concrete(), move |grad, out| {
            (grad * out)
                .into_concrete()
                .mul_scalar(T::from_f32(std::f32::consts::LN_2))
                .into_concrete()
        })
    }

    pub fn less_approximate_exp(&self) -> Self {
        self.unary_from_value(
            self.value.less_approximate_exp().into_concrete(),
            move |grad, out| (grad * out).into_concrete(),
        )
    }

    pub fn log2(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.log2().into_concrete(), move |grad, _| {
            (grad / input.clone())
                .into_concrete()
                .div_scalar(T::from_f32(std::f32::consts::LN_2))
                .into_concrete()
        })
    }

    pub fn sin(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sin().into_concrete(), move |grad, _| {
            (grad * input.cos().into_concrete()).into_concrete()
        })
    }

    pub fn sinh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sinh().into_concrete(), move |grad, _| {
            (grad * input.cosh().into_concrete()).into_concrete()
        })
    }

    pub fn tan(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.tan().into_concrete(), move |grad, _| {
            let cos = input.cos().into_concrete();
            (grad / (cos.clone() * cos).into_concrete()).into_concrete()
        })
    }

    pub fn tanh_exact(&self) -> Self {
        self.unary_from_value(self.value.tanh_exact().into_concrete(), move |grad, out| {
            let one_minus_sq = (RawTensor::splat(&out.device(), T::from_f32(1.0), out.shape())
                - out.sqr().into_concrete())
            .into_concrete();
            (grad * one_minus_sq).into_concrete()
        })
    }

    /// Cast the raw value to another element type, dropping the tape.
    pub fn cast_raw<D2>(&self) -> crate::Tensor<R, D2>
    where
        T: crate::CastTo<D2> + crate::CastTensor<D2>,
        D2: crate::SimdElement + crate::DataType + Default,
    {
        self.value.cast()
    }

    /// Differentiable cast between autograd element types: the forward casts
    /// the value, the backward casts the gradient back. This is the bridge for
    /// mixed-precision training (e.g. f32 master weights feeding an f16 model).
    pub fn cast<T2>(&self) -> Tensor<R, T2>
    where
        T2: AutogradElement,
        T: crate::CastElement<T2>,
        T2: crate::CastElement<T>,
        crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T2>,
    {
        let value = self.value.cast::<T2>();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R, T2>(&*gradient, "cast")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.cast::<T>()),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn to_concrete(&self) -> Self {
        self.unary_from_value(self.value.to_concrete(), move |grad, _| grad)
    }

    pub fn relu(&self) -> Self {
        self.max_elementwise(0.0)
    }

    pub fn clamp(&self, min: f32, max: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(
            self.value
                .clamp(T::from_f32(min), T::from_f32(max))
                .into_concrete(),
            move |grad, _| {
                let lower = input.mt(T::from_f32(min)).into_concrete();
                let upper = input.lt(T::from_f32(max)).into_concrete();
                ((grad * lower).into_concrete() * upper).into_concrete()
            },
        )
    }

    pub fn eq(&self, rhs: f32) -> Self {
        self.unary_from_value(
            self.value.eq(T::from_f32(rhs)).into_concrete(),
            move |_, out| RawTensor::zeros(&out.device(), out.shape()),
        )
    }

    pub fn eq_scalar(&self, rhs: f32) -> Self {
        self.eq(rhs)
    }

    pub fn eq_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.eq_tensor(&rhs.value).into_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn gt_scalar(&self, rhs: f32) -> Self {
        self.unary_from_value(
            self.value.gt_scalar(T::from_f32(rhs)).into_concrete(),
            move |_, out| RawTensor::zeros(&out.device(), out.shape()),
        )
    }

    pub fn gt_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.gt_tensor(&rhs.value).into_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn gte_scalar(&self, rhs: f32) -> Self {
        self.unary_from_value(
            self.value.gte_scalar(T::from_f32(rhs)).into_concrete(),
            move |_, out| RawTensor::zeros(&out.device(), out.shape()),
        )
    }

    pub fn gte_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.gte_tensor(&rhs.value).into_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn lt(&self, rhs: f32) -> Self {
        self.unary_from_value(
            self.value.lt(T::from_f32(rhs)).into_concrete(),
            move |_, out| RawTensor::zeros(&out.device(), out.shape()),
        )
    }

    pub fn lt_scalar(&self, rhs: f32) -> Self {
        self.lt(rhs)
    }

    pub fn lt_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.lt_tensor(&rhs.value).into_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn lte(&self, rhs: f32) -> Self {
        self.unary_from_value(
            self.value.lte(T::from_f32(rhs)).into_concrete(),
            move |_, out| RawTensor::zeros(&out.device(), out.shape()),
        )
    }

    pub fn lte_scalar(&self, rhs: f32) -> Self {
        self.lte(rhs)
    }

    pub fn lte_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.lte_tensor(&rhs.value).into_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn max_elementwise(&self, rhs: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(
            self.value.max_elementwise(T::from_f32(rhs)).into_concrete(),
            move |grad, _| (grad * input.mt(T::from_f32(rhs)).into_concrete()).into_concrete(),
        )
    }

    pub fn max_scalar(&self, rhs: f32) -> Self {
        self.max_elementwise(rhs)
    }

    pub fn min_elementwise(&self, rhs: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(
            self.value.min_elementwise(T::from_f32(rhs)).into_concrete(),
            move |grad, _| (grad * input.lt(T::from_f32(rhs)).into_concrete()).into_concrete(),
        )
    }

    pub fn min_scalar(&self, rhs: f32) -> Self {
        self.min_elementwise(rhs)
    }

    pub fn mt(&self, rhs: f32) -> Self {
        self.gt_scalar(rhs)
    }

    pub fn mte(&self, rhs: f32) -> Self {
        self.gte_scalar(rhs)
    }

    pub fn ne(&self, rhs: f32) -> Self {
        self.unary_from_value(
            self.value.ne(T::from_f32(rhs)).into_concrete(),
            move |_, out| RawTensor::zeros(&out.device(), out.shape()),
        )
    }

    pub fn ne_scalar(&self, rhs: f32) -> Self {
        self.ne(rhs)
    }

    pub fn ne_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.ne_tensor(&rhs.value).into_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn sigmoid(&self) -> Self {
        self.mul_scalar(-1.0).exp().add_scalar(1.0).pow_scalar(-1.0)
    }

    pub fn silu(&self) -> Self {
        let denom = self.mul_scalar(-1.0).exp().add_scalar(1.0);
        self.div(&denom)
    }

    pub fn gelu(&self) -> Self {
        // Fused forward (single elementwise chain) with an analytic backward:
        // gelu'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * c * (1 + 3 * 0.044715 * x^2)
        // where t = tanh(c * (x + 0.044715 * x^3)) and c = sqrt(2 / pi).
        let value = self.value.gelu();
        let input_value = self.value.clone();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let grad = downcast_tensor::<R, T>(&*gradient, "gelu")?;
            let coeff = T::from_f32((2.0f32 / std::f32::consts::PI).sqrt());
            let one = T::from_f32(1.0);
            let x = input_value.to_concrete();
            let x_sq = x.sqr().into_concrete();
            let inner_factor = (&x_sq * T::from_f32(0.044_715) + one).into_concrete();
            let inner = ((&x * &inner_factor).into_concrete() * coeff).into_concrete();
            let t = inner.tanh().into_concrete();
            let sech_sq = (t.sqr() * T::from_f32(-1.0) + one).into_concrete();
            let du = ((&x_sq * T::from_f32(3.0 * 0.044_715) + one).into_concrete() * coeff)
                .into_concrete();
            let tail = ((&x * &sech_sq).into_concrete() * du).into_concrete();
            let dgelu = (((t + one).into_concrete() + tail).into_concrete() * T::from_f32(0.5))
                .into_concrete();
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new((&grad * &dgelu).into_concrete()),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn tanh(&self) -> Self {
        self.unary_from_value(self.value.tanh().into_concrete(), move |grad, out| {
            let one_minus_sq = (RawTensor::splat(&out.device(), T::from_f32(1.0), out.shape())
                - out.sqr().into_concrete())
            .into_concrete();
            (grad * one_minus_sq).into_concrete()
        })
    }

    pub fn exp(&self) -> Self {
        self.unary_from_value(self.value.exp().into_concrete(), move |grad, out| {
            (grad * out).into_concrete()
        })
    }

    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Self {
        assert_same_graph(self, on_true);
        assert_same_graph(self, on_false);

        let value = self
            .value
            .where_cond(&on_true.value, &on_false.value)
            .into_concrete();
        let condition_id = self.handle.id;
        let true_id = on_true.handle.id;
        let false_id = on_false.handle.id;
        let condition = self.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R, T>(&*gradient, "where_cond")?;
            let zeros = RawTensor::zeros(&condition.device(), condition.shape());
            let ones = RawTensor::splat(&condition.device(), T::from_f32(1.0), condition.shape());
            let true_mask = condition.where_cond(&ones, &zeros).into_concrete();
            let false_mask = condition.where_cond(&zeros, &ones).into_concrete();
            Ok(vec![
                BackwardTarget {
                    node: condition_id,
                    gradient: Box::new(zeros),
                },
                BackwardTarget {
                    node: true_id,
                    gradient: Box::new((gradient.clone() * true_mask).into_concrete()),
                },
                BackwardTarget {
                    node: false_id,
                    gradient: Box::new((gradient * false_mask).into_concrete()),
                },
            ])
        });
        self.emit_op(
            value,
            vec![
                self.handle.clone(),
                on_true.handle.clone(),
                on_false.handle.clone(),
            ],
            Some(backward),
        )
    }

    pub fn log(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.log().into_concrete(), move |grad, _| {
            (grad / input.clone()).into_concrete()
        })
    }

    pub fn sqrt(&self) -> Self {
        self.unary_from_value(self.value.sqrt().into_concrete(), move |grad, out| {
            let denom = out.mul_scalar(T::from_f32(2.0)).into_concrete();
            (grad / denom).into_concrete()
        })
    }
}

macro_rules! impl_autograd_pairwise_op {
    ($trait:ident, $method:ident) => {
        impl<const R: usize, T: AutogradElement> std::ops::$trait<Tensor<R, T>> for Tensor<R, T>
        where
            crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
        {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: Tensor<R, T>) -> Tensor<R, T> {
                Tensor::$method(&self, &rhs)
            }
        }

        impl<const R: usize, T: AutogradElement> std::ops::$trait<&Tensor<R, T>> for Tensor<R, T>
        where
            crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
        {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: &Tensor<R, T>) -> Tensor<R, T> {
                Tensor::$method(&self, rhs)
            }
        }

        impl<const R: usize, T: AutogradElement> std::ops::$trait<Tensor<R, T>> for &Tensor<R, T>
        where
            crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
        {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: Tensor<R, T>) -> Tensor<R, T> {
                Tensor::$method(self, &rhs)
            }
        }

        impl<const R: usize, T: AutogradElement> std::ops::$trait<&Tensor<R, T>> for &Tensor<R, T>
        where
            crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
        {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: &Tensor<R, T>) -> Tensor<R, T> {
                Tensor::$method(self, rhs)
            }
        }
    };
}

impl_autograd_pairwise_op!(Add, add);
impl_autograd_pairwise_op!(Sub, sub);
impl_autograd_pairwise_op!(Mul, mul);
impl_autograd_pairwise_op!(Div, div);

macro_rules! impl_autograd_scalar_op {
    ($trait:ident, $method:ident, $scalar_method:ident) => {
        impl<const R: usize, T: AutogradElement> std::ops::$trait<f32> for Tensor<R, T>
        where
            crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
        {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: f32) -> Tensor<R, T> {
                Tensor::$scalar_method(&self, rhs)
            }
        }

        impl<const R: usize, T: AutogradElement> std::ops::$trait<f32> for &Tensor<R, T>
        where
            crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
        {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: f32) -> Tensor<R, T> {
                Tensor::$scalar_method(self, rhs)
            }
        }
    };
}

impl_autograd_scalar_op!(Mul, mul, mul_scalar);
impl_autograd_scalar_op!(Add, add, add_scalar);
impl_autograd_scalar_op!(Sub, sub, sub_scalar);
impl_autograd_scalar_op!(Div, div, div_scalar);

impl<const R: usize, T: AutogradElement> std::ops::Neg for Tensor<R, T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
{
    type Output = Tensor<R, T>;

    fn neg(self) -> Tensor<R, T> {
        Tensor::neg(&self)
    }
}

impl<const R: usize, T: AutogradElement> std::ops::Neg for &Tensor<R, T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
{
    type Output = Tensor<R, T>;

    fn neg(self) -> Tensor<R, T> {
        Tensor::neg(self)
    }
}
