use super::*;

impl<const R: usize> Tensor<R> {
    pub fn add(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() + rhs.value.clone()).to_concrete(),
            |grad, _, _| vec![grad.clone().to_concrete(), grad.to_concrete()],
        )
    }

    pub fn add_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.add(&rhs)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() - rhs.value.clone()).to_concrete(),
            |grad, _, _| vec![grad.clone().to_concrete(), (-grad).to_concrete()],
        )
    }

    pub fn sub_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.sub(&rhs)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() * rhs.value.clone()).to_concrete(),
            |grad, lhs, rhs| {
                vec![
                    (grad.clone() * rhs).to_concrete(),
                    (grad * lhs).to_concrete(),
                ]
            },
        )
    }

    pub fn mul_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.mul(&rhs)
    }

    pub fn div(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() / rhs.value.clone()).to_concrete(),
            |grad, lhs, rhs| {
                let lhs_grad = (grad.clone() / rhs.clone()).to_concrete();
                let rhs_grad = (-((grad * lhs) / rhs.sqr().to_concrete())).to_concrete();
                vec![lhs_grad, rhs_grad]
            },
        )
    }

    pub fn div_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.div(&rhs)
    }

    pub fn pow(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            self.value.pow(&rhs.value).to_concrete(),
            |grad, lhs, rhs| {
                let rhs_minus_one = rhs.sub_scalar(1.0).to_concrete();
                let lhs_power = lhs.pow(&rhs_minus_one).to_concrete();
                let lhs_grad =
                    ((grad.clone() * rhs.clone()).to_concrete() * lhs_power).to_concrete();
                let rhs_grad = ((grad * lhs.pow(&rhs).to_concrete()).to_concrete()
                    * lhs.log().to_concrete())
                .to_concrete();
                vec![lhs_grad, rhs_grad]
            },
        )
    }

    pub fn pow_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2>) -> Tensor<R3> {
        let out_shape: [usize; R3] =
            crate::composite::broadcast_shapes(&self.shape(), &second.shape());
        let lhs = self.broadcast_as(out_shape);
        let rhs = second.broadcast_as(out_shape);
        lhs.pow(&rhs)
    }

    pub fn pow_elementwise(&self, exponent: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(
            self.value.pow_elementwise(exponent).to_concrete(),
            move |grad, _| {
                let power = input.pow_elementwise(exponent - 1.0).to_concrete();
                (grad * power)
                    .to_concrete()
                    .mul_scalar(exponent)
                    .to_concrete()
            },
        )
    }

    pub fn pow_scalar(&self, exponent: f32) -> Self {
        self.pow_elementwise(exponent)
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(self.value.add_scalar(scalar), move |grad, _| grad)
    }

    pub fn sub_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(self.value.sub_scalar(scalar), move |grad, _| grad)
    }

    pub fn mul_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.mul_scalar(scalar).to_concrete(),
            move |grad, _| grad.mul_scalar(scalar).to_concrete(),
        )
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.div_scalar(scalar).to_concrete(),
            move |grad, _| grad.div_scalar(scalar).to_concrete(),
        )
    }

    pub fn neg(&self) -> Self {
        self.unary_from_value((-self.value.clone()).to_concrete(), move |grad, _| {
            (-grad).to_concrete()
        })
    }

    pub fn sqr(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sqr().to_concrete(), move |grad, _| {
            ((grad * input.clone()).to_concrete().mul_scalar(2.0)).to_concrete()
        })
    }

    pub fn abs(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.abs().to_concrete(), move |grad, _| {
            let positive = input.mt(0.0).to_concrete();
            let negative = input.lt(0.0).to_concrete();
            ((grad.clone() * positive).to_concrete() - (grad * negative).to_concrete())
                .to_concrete()
        })
    }

    pub fn acos(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.acos().to_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), 1.0, input.shape())
                - input.sqr().to_concrete())
            .to_concrete()
            .sqrt()
            .to_concrete();
            (-(grad / denom).to_concrete()).to_concrete()
        })
    }

    pub fn acosh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.acosh().to_concrete(), move |grad, _| {
            let lower = input.add_scalar(-1.0).to_concrete().sqrt().to_concrete();
            let upper = input.add_scalar(1.0).to_concrete().sqrt().to_concrete();
            (grad / (lower * upper).to_concrete()).to_concrete()
        })
    }

    pub fn approximate_exp(&self) -> Self {
        self.unary_from_value(
            self.value.approximate_exp().to_concrete(),
            move |grad, out| (grad * out).to_concrete(),
        )
    }

    pub fn asin(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.asin().to_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), 1.0, input.shape())
                - input.sqr().to_concrete())
            .to_concrete()
            .sqrt()
            .to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn asinh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.asinh().to_concrete(), move |grad, _| {
            let denom = input
                .sqr()
                .add_scalar(1.0)
                .to_concrete()
                .sqrt()
                .to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn atan(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.atan().to_concrete(), move |grad, _| {
            let denom = input.sqr().add_scalar(1.0).to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn atanh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.atanh().to_concrete(), move |grad, _| {
            let denom = (RawTensor::splat(&input.device(), 1.0, input.shape())
                - input.sqr().to_concrete())
            .to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn cos(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.cos().to_concrete(), move |grad, _| {
            (-(grad * input.sin().to_concrete()).to_concrete()).to_concrete()
        })
    }

    pub fn cosh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.cosh().to_concrete(), move |grad, _| {
            (grad * input.sinh().to_concrete()).to_concrete()
        })
    }

    pub fn exp2(&self) -> Self {
        self.unary_from_value(self.value.exp2().to_concrete(), move |grad, out| {
            (grad * out)
                .to_concrete()
                .mul_scalar(std::f32::consts::LN_2)
                .to_concrete()
        })
    }

    pub fn less_approximate_exp(&self) -> Self {
        self.unary_from_value(
            self.value.less_approximate_exp().to_concrete(),
            move |grad, out| (grad * out).to_concrete(),
        )
    }

    pub fn log2(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.log2().to_concrete(), move |grad, _| {
            (grad / input.clone())
                .to_concrete()
                .div_scalar(std::f32::consts::LN_2)
                .to_concrete()
        })
    }

    pub fn sin(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sin().to_concrete(), move |grad, _| {
            (grad * input.cos().to_concrete()).to_concrete()
        })
    }

    pub fn sinh(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sinh().to_concrete(), move |grad, _| {
            (grad * input.cosh().to_concrete()).to_concrete()
        })
    }

    pub fn tan(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.tan().to_concrete(), move |grad, _| {
            let cos = input.cos().to_concrete();
            (grad / (cos.clone() * cos).to_concrete()).to_concrete()
        })
    }

    pub fn tanh_exact(&self) -> Self {
        self.unary_from_value(self.value.tanh_exact().to_concrete(), move |grad, out| {
            let one_minus_sq = (RawTensor::splat(&out.device(), 1.0, out.shape())
                - out.sqr().to_concrete())
            .to_concrete();
            (grad * one_minus_sq).to_concrete()
        })
    }

    pub fn cast<D2>(&self) -> crate::Tensor<R, D2>
    where
        f32: crate::CastTo<D2> + crate::CastTensor<D2>,
        D2: crate::SimdElement + crate::DataType + Default,
    {
        self.value.cast()
    }

    pub fn to_concrete(&self) -> Self {
        self.unary_from_value(self.value.to_concrete(), move |grad, _| grad)
    }

    pub fn relu(&self) -> Self {
        self.max_elementwise(0.0)
    }

    pub fn clamp(&self, min: f32, max: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.clamp(min, max).to_concrete(), move |grad, _| {
            let lower = input.mt(min).to_concrete();
            let upper = input.lt(max).to_concrete();
            ((grad * lower).to_concrete() * upper).to_concrete()
        })
    }

    pub fn eq(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.eq(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn eq_scalar(&self, rhs: f32) -> Self {
        self.eq(rhs)
    }

    pub fn eq_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.eq_tensor(&rhs.value).to_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn gt_scalar(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.gt_scalar(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn gt_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.gt_tensor(&rhs.value).to_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn gte_scalar(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.gte_scalar(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn gte_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.gte_tensor(&rhs.value).to_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn lt(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.lt(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn lt_scalar(&self, rhs: f32) -> Self {
        self.lt(rhs)
    }

    pub fn lt_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.lt_tensor(&rhs.value).to_concrete(),
            move |_, lhs, rhs| {
                vec![
                    RawTensor::zeros(&lhs.device(), lhs.shape()),
                    RawTensor::zeros(&rhs.device(), rhs.shape()),
                ]
            },
        )
    }

    pub fn lte(&self, rhs: f32) -> Self {
        self.unary_from_value(self.value.lte(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn lte_scalar(&self, rhs: f32) -> Self {
        self.lte(rhs)
    }

    pub fn lte_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.lte_tensor(&rhs.value).to_concrete(),
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
            self.value.max_elementwise(rhs).to_concrete(),
            move |grad, _| (grad * input.mt(rhs).to_concrete()).to_concrete(),
        )
    }

    pub fn max_scalar(&self, rhs: f32) -> Self {
        self.max_elementwise(rhs)
    }

    pub fn min_elementwise(&self, rhs: f32) -> Self {
        let input = self.value.clone();
        self.unary_from_value(
            self.value.min_elementwise(rhs).to_concrete(),
            move |grad, _| (grad * input.lt(rhs).to_concrete()).to_concrete(),
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
        self.unary_from_value(self.value.ne(rhs).to_concrete(), move |_, out| {
            RawTensor::zeros(&out.device(), out.shape())
        })
    }

    pub fn ne_scalar(&self, rhs: f32) -> Self {
        self.ne(rhs)
    }

    pub fn ne_tensor(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        self.binary_op(
            rhs,
            self.value.ne_tensor(&rhs.value).to_concrete(),
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
        let cubic = self.sqr().mul(self);
        let inner = self
            .add(&cubic.mul_scalar(0.044_715))
            .mul_scalar((2.0 / std::f32::consts::PI).sqrt());
        let gate = inner.tanh().add_scalar(1.0);
        self.mul(&gate).mul_scalar(0.5)
    }

    pub fn tanh(&self) -> Self {
        self.unary_from_value(self.value.tanh().to_concrete(), move |grad, out| {
            let one_minus_sq = (RawTensor::splat(&out.device(), 1.0, out.shape())
                - out.sqr().to_concrete())
            .to_concrete();
            (grad * one_minus_sq).to_concrete()
        })
    }

    pub fn exp(&self) -> Self {
        self.unary_from_value(self.value.exp().to_concrete(), move |grad, out| {
            (grad * out).to_concrete()
        })
    }

    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Self {
        assert_same_graph(self, on_true);
        assert_same_graph(self, on_false);

        let value = self
            .value
            .where_cond(&on_true.value, &on_false.value)
            .to_concrete();
        let condition_id = self.handle.id;
        let true_id = on_true.handle.id;
        let false_id = on_false.handle.id;
        let condition = self.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "where_cond")?;
            let zeros = RawTensor::zeros(&condition.device(), condition.shape());
            let ones = RawTensor::splat(&condition.device(), 1.0, condition.shape());
            let true_mask = condition.where_cond(&ones, &zeros).to_concrete();
            let false_mask = condition.where_cond(&zeros, &ones).to_concrete();
            Ok(vec![
                BackwardTarget {
                    node: condition_id,
                    gradient: Box::new(zeros),
                },
                BackwardTarget {
                    node: true_id,
                    gradient: Box::new((gradient.clone() * true_mask).to_concrete()),
                },
                BackwardTarget {
                    node: false_id,
                    gradient: Box::new((gradient * false_mask).to_concrete()),
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
        self.unary_from_value(self.value.log().to_concrete(), move |grad, _| {
            (grad / input.clone()).to_concrete()
        })
    }

    pub fn sqrt(&self) -> Self {
        self.unary_from_value(self.value.sqrt().to_concrete(), move |grad, out| {
            let denom = out.mul_scalar(2.0).to_concrete();
            (grad / denom).to_concrete()
        })
    }
}

macro_rules! impl_autograd_pairwise_op {
    ($trait:ident, $method:ident) => {
        impl<const R: usize> std::ops::$trait<Tensor<R>> for Tensor<R> {
            type Output = Tensor<R>;

            fn $method(self, rhs: Tensor<R>) -> Tensor<R> {
                Tensor::$method(&self, &rhs)
            }
        }

        impl<const R: usize> std::ops::$trait<&Tensor<R>> for Tensor<R> {
            type Output = Tensor<R>;

            fn $method(self, rhs: &Tensor<R>) -> Tensor<R> {
                Tensor::$method(&self, rhs)
            }
        }

        impl<const R: usize> std::ops::$trait<Tensor<R>> for &Tensor<R> {
            type Output = Tensor<R>;

            fn $method(self, rhs: Tensor<R>) -> Tensor<R> {
                Tensor::$method(self, &rhs)
            }
        }

        impl<const R: usize> std::ops::$trait<&Tensor<R>> for &Tensor<R> {
            type Output = Tensor<R>;

            fn $method(self, rhs: &Tensor<R>) -> Tensor<R> {
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
        impl<const R: usize> std::ops::$trait<f32> for Tensor<R> {
            type Output = Tensor<R>;

            fn $method(self, rhs: f32) -> Tensor<R> {
                Tensor::$scalar_method(&self, rhs)
            }
        }

        impl<const R: usize> std::ops::$trait<f32> for &Tensor<R> {
            type Output = Tensor<R>;

            fn $method(self, rhs: f32) -> Tensor<R> {
                Tensor::$scalar_method(self, rhs)
            }
        }
    };
}

impl_autograd_scalar_op!(Mul, mul, mul_scalar);
impl_autograd_scalar_op!(Add, add, add_scalar);
impl_autograd_scalar_op!(Sub, sub, sub_scalar);
impl_autograd_scalar_op!(Div, div, div_scalar);

impl<const R: usize> std::ops::Neg for Tensor<R> {
    type Output = Tensor<R>;

    fn neg(self) -> Tensor<R> {
        Tensor::neg(&self)
    }
}

impl<const R: usize> std::ops::Neg for &Tensor<R> {
    type Output = Tensor<R>;

    fn neg(self) -> Tensor<R> {
        Tensor::neg(self)
    }
}

