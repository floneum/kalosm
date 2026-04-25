use crate::sgemm::SgemmParams;

#[inline]
pub fn gemm_parameters(m: usize, n: usize, k: usize) -> SgemmParams {
    let m_u = m;
    let n_u = n;
    let k_u = k;
    let m = m as f32;
    let n = n as f32;
    let k = k as f32;
    #[inline]
    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let t = b;
            b = a % t;
            a = t;
        }
        a
    }
    let m_over_n = m / n;
    let m_over_k = m / k;
    let n_over_m = n / m;
    let n_over_k = n / k;
    let k_over_m = k / m;
    let k_over_n = k / n;
    let log2_m = m.log2();
    let log2_n = n.log2();
    let log2_k = k.log2();
    let sum_dim = m + n + k;
    let max_dim = m.max(n).max(k);
    let min_dim = m.min(n).min(k);
    let diff_mn = (m - n).abs();
    let diff_nk = (n - k).abs();
    let diff_mk = (m - k).abs();
    let m_eq_k = if m_u == k_u { 1.0 } else { 0.0 };
    let gcd_mn = gcd(m_u, n_u) as f32;
    let gcd_nk = gcd(n_u, k_u) as f32;
    let gcd_mk = gcd(m_u, k_u) as f32;

    // Nested decisions from the pruned tree:
    if sum_dim <= 1472f32 {
        if log2_k <= 8.5f32 {
            if n_over_k <= 1.5f32 {
                if sum_dim <= 448f32 {
                    SgemmParams::new(false, 32u32, 32u32, 32u32, 2u32, 2u32)
                } else if m_over_n <= 0.75f32 {
                    SgemmParams::new(true, 16u32, 64u32, 32u32, 2u32, 2u32)
                } else {
                    SgemmParams::new(true, 32u32, 32u32, 32u32, 2u32, 2u32)
                }
            } else if gcd_mn <= 384f32 {
                if m_eq_k <= 0.5f32 {
                    if diff_mn <= 320f32 {
                        if m_over_n <= 1.5f32 {
                            SgemmParams::new(true, 32u32, 32u32, 32u32, 2u32, 2u32)
                        } else {
                            SgemmParams::new(false, 32u32, 32u32, 32u32, 2u32, 2u32)
                        }
                    } else if log2_m <= 7.5f32 {
                        if diff_nk <= 512f32 {
                            SgemmParams::new(true, 64u32, 64u32, 32u32, 4u32, 4u32)
                        } else {
                            SgemmParams::new(true, 32u32, 32u32, 32u32, 2u32, 2u32)
                        }
                    } else {
                        SgemmParams::new(false, 32u32, 32u32, 16u32, 2u32, 2u32)
                    }
                } else if sum_dim <= 896f32 {
                    if n_over_m <= 3f32 {
                        SgemmParams::new(false, 32u32, 32u32, 16u32, 2u32, 2u32)
                    } else {
                        SgemmParams::new(false, 32u32, 16u32, 32u32, 2u32, 2u32)
                    }
                } else if k <= 192f32 {
                    SgemmParams::new(false, 16u32, 32u32, 8u32, 2u32, 2u32)
                } else {
                    SgemmParams::new(false, 32u32, 32u32, 32u32, 2u32, 2u32)
                }
            } else if k_over_n <= 0.375f32 {
                SgemmParams::new(true, 128u32, 16u32, 8u32, 4u32, 4u32)
            } else {
                SgemmParams::new(false, 64u32, 8u32, 16u32, 2u32, 2u32)
            }
        } else if gcd_nk <= 192f32 {
            if k_over_m <= 3f32 {
                if n_over_m <= 0.375f32 {
                    SgemmParams::new(false, 32u32, 16u32, 32u32, 2u32, 2u32)
                } else {
                    SgemmParams::new(true, 32u32, 32u32, 32u32, 2u32, 2u32)
                }
            } else if log2_m <= 7.5f32 {
                if max_dim <= 768f32 {
                    SgemmParams::new(true, 32u32, 32u32, 64u32, 2u32, 2u32)
                } else {
                    SgemmParams::new(false, 64u32, 16u32, 32u32, 2u32, 2u32)
                }
            } else {
                SgemmParams::new(false, 32u32, 16u32, 64u32, 2u32, 2u32)
            }
        } else if diff_mn <= 192f32 {
            if diff_mk <= 320f32 {
                SgemmParams::new(false, 16u32, 32u32, 32u32, 2u32, 2u32)
            } else if k_over_m <= 6f32 {
                SgemmParams::new(true, 64u32, 16u32, 16u32, 2u32, 2u32)
            } else {
                SgemmParams::new(true, 8u32, 32u32, 32u32, 2u32, 2u32)
            }
        } else {
            SgemmParams::new(false, 64u32, 8u32, 16u32, 2u32, 2u32)
        }
    } else if sum_dim <= 3264f32 {
        if min_dim <= 384f32 {
            if log2_m <= 7.5f32 {
                if m_over_n <= 0.1875f32 {
                    if n_over_k <= 12f32 {
                        if sum_dim <= 2944f32 || m_over_k <= 0.09375f32 {
                            SgemmParams::new(false, 64u32, 8u32, 16u32, 2u32, 2u32)
                        } else {
                            SgemmParams::new(false, 64u32, 32u32, 4u32, 4u32, 4u32)
                        }
                    } else {
                        SgemmParams::new(false, 32u32, 32u32, 16u32, 2u32, 2u32)
                    }
                } else if n <= 384f32 {
                    if diff_nk <= 1856f32 {
                        SgemmParams::new(true, 32u32, 16u32, 32u32, 2u32, 2u32)
                    } else {
                        SgemmParams::new(true, 16u32, 64u32, 64u32, 2u32, 2u32)
                    }
                } else if m_over_k <= 0.09375f32 {
                    SgemmParams::new(true, 32u32, 32u32, 32u32, 2u32, 2u32)
                } else {
                    SgemmParams::new(true, 64u32, 8u32, 16u32, 2u32, 2u32)
                }
            } else if m_over_n <= 12f32 {
                if gcd_nk <= 768f32 {
                    if gcd_mk <= 768f32 || min_dim <= 192f32 {
                        SgemmParams::new(false, 64u32, 8u32, 16u32, 2u32, 2u32)
                    } else {
                        SgemmParams::new(false, 128u32, 16u32, 8u32, 4u32, 4u32)
                    }
                } else {
                    SgemmParams::new(false, 64u32, 32u32, 4u32, 4u32, 4u32)
                }
            } else if n_over_k <= 0.375f32 {
                if k <= 768f32 {
                    SgemmParams::new(false, 64u32, 32u32, 8u32, 4u32, 4u32)
                } else {
                    SgemmParams::new(false, 64u32, 32u32, 4u32, 4u32, 4u32)
                }
            } else if gcd_mk <= 192f32 {
                SgemmParams::new(true, 16u32, 32u32, 16u32, 2u32, 2u32)
            } else {
                SgemmParams::new(false, 64u32, 8u32, 16u32, 2u32, 2u32)
            }
        } else if log2_k <= 9.5f32 {
            if sum_dim <= 2304f32 {
                SgemmParams::new(false, 64u32, 8u32, 16u32, 2u32, 2u32)
            } else if log2_n <= 10.5f32 {
                SgemmParams::new(false, 64u32, 32u32, 4u32, 4u32, 4u32)
            } else {
                SgemmParams::new(false, 128u32, 16u32, 8u32, 4u32, 4u32)
            }
        } else if sum_dim <= 2816f32 {
            if diff_mn <= 256f32 {
                SgemmParams::new(false, 64u32, 32u32, 4u32, 4u32, 4u32)
            } else {
                SgemmParams::new(false, 64u32, 16u32, 16u32, 2u32, 2u32)
            }
        } else {
            SgemmParams::new(false, 16u32, 128u32, 8u32, 4u32, 4u32)
        }
    } else if min_dim <= 768f32 {
        if k <= 384f32 {
            if log2_m <= 10.5f32 || k <= 192f32 {
                SgemmParams::new(false, 64u32, 8u32, 16u32, 2u32, 2u32)
            } else {
                SgemmParams::new(false, 128u32, 16u32, 8u32, 4u32, 4u32)
            }
        } else if sum_dim <= 4288f32 {
            if diff_mn <= 640f32 {
                SgemmParams::new(false, 64u32, 16u32, 32u32, 2u32, 2u32)
            } else if diff_nk <= 896f32 {
                if k_over_n <= 1.5f32 {
                    SgemmParams::new(false, 16u32, 128u32, 8u32, 4u32, 4u32)
                } else if m_over_n <= 6f32 {
                    SgemmParams::new(false, 64u32, 32u32, 8u32, 4u32, 4u32)
                } else {
                    SgemmParams::new(false, 64u32, 16u32, 16u32, 2u32, 2u32)
                }
            } else {
                SgemmParams::new(false, 16u32, 128u32, 8u32, 4u32, 4u32)
            }
        } else if m <= 1280f32 {
            if m_over_k <= 0.1875f32 {
                SgemmParams::new(false, 64u32, 16u32, 32u32, 2u32, 2u32)
            } else {
                SgemmParams::new(false, 32u32, 32u32, 16u32, 4u32, 4u32)
            }
        } else if n_over_m <= 0.1875f32 {
            SgemmParams::new(false, 32u32, 32u32, 8u32, 4u32, 4u32)
        } else if log2_n <= 10f32 {
            SgemmParams::new(false, 64u32, 32u32, 16u32, 4u32, 4u32)
        } else {
            SgemmParams::new(false, 32u32, 128u32, 8u32, 4u32, 4u32)
        }
    } else {
        // Large matrices (sum_dim > 3264, min_dim > 768):
        // Use 64 threads, small shared memory, high work-per-thread for Apple Silicon occupancy
        SgemmParams::new(false, 48u32, 32u32, 8u32, 6u32, 4u32)
    }
}
