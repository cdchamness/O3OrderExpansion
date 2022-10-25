use std::ops::Mul;
use crate::bra_ket::*;
use crate::term::*;

#[derive(Clone)]
pub struct KDeltas {
    index_type_1: char,
    index_type_2: char,
    shift_1: Vec<i8>,
    shift_2: Vec<i8>,
}
#[derive(Clone)]
pub struct InnerProduct {
    scalar: f64,
    bra: BraKet,
    inner: String,
    ket: BraKet,
    x_deltas: Vec<KDeltas>,
    mu_deltas: Vec<KDeltas>,
}

impl InnerProduct {

    pub fn new(scalar: f64, bra: BraKet, inner: String, ket: BraKet, x_deltas: Vec<KDeltas>, mu_deltas: Vec<KDeltas>) -> InnerProduct {
        InnerProduct {
            scalar,
            bra,
            inner,
            ket,
            x_deltas,
            mu_deltas,
        }
    }
}

impl Mul<InnerProduct> for InnerProduct {
    type Output = Term;

    fn mul(self, rhs: Self) -> Term {
        Term::new(vec![self, rhs])
    }
}

impl Mul<Term> for InnerProduct {
    type Output = Term;

    fn mul(self, rhs: Term) -> Term {
        rhs * self
    }
}

impl Mul<f64> for InnerProduct {
    type Output = Self;

    fn mul(self, rhs: f64) -> InnerProduct {
        let mut dup = self.clone();
        dup.scalar *= rhs;
        dup
    }
}

impl Mul<Vec<InnerProduct>> for InnerProduct {
    type Output = Vec<Term>;

    fn mul(self, rhs: Vec<InnerProduct>) -> Vec<Term> {
        let mut out = Vec::new();
        for item in rhs {
            let dup = self.clone();
            out.push(dup * item);
        }
        out
    }
}

