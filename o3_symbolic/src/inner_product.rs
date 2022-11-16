use std::cmp::{Ord, Ordering, PartialOrd};
use std::fmt;
use std::ops::{Mul, MulAssign};

use crate::bra_ket::*;
use crate::kdelta::*;
use crate::term::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gen {
    alpha_type: char,
}

impl Gen {
    pub fn new(alpha_type: char) -> Gen {
        Gen { alpha_type }
    }

    pub fn get_type(&self) -> char {
        self.alpha_type
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InnerProduct {
    scalar: f64,
    bra: BraKet,
    inner: Vec<Gen>,
    ket: BraKet,
    delta: Option<KDelta>,
}

impl InnerProduct {
    pub fn new(
        scalar: f64,
        bra: BraKet,
        inner: Vec<Gen>,
        ket: BraKet,
        delta: Option<KDelta>,
    ) -> InnerProduct {
        InnerProduct {
            scalar,
            bra,
            inner,
            ket,
            delta,
        }
    }

    pub fn get_scalar(&self) -> f64 {
        self.scalar
    }

    pub fn get_bra(&self) -> BraKet {
        self.bra.clone()
    }

    pub fn get_inner(&self) -> Vec<Gen> {
        self.inner.clone()
    }

    pub fn get_ket(&self) -> BraKet {
        self.ket.clone()
    }

    pub fn get_delta(&self) -> Option<KDelta> {
        self.delta.clone()
    }

    pub fn extract_scalar(&mut self) -> f64 {
        let num = self.scalar;
        self.scalar = 1.0;
        num
    }

    pub fn set_scalar(&mut self, num: f64) {
        self.scalar = num;
    }

    pub fn clear_inner_alpha(&mut self, alpha: char) {
        let mut new_inner = Vec::new();
        for gen in &self.inner {
            if gen.get_type() != alpha {
                new_inner.push(gen.clone());
            }
        }
        self.inner = new_inner;
    }

    pub fn add_to_inner(&mut self, gen: Gen) {
        self.inner.push(gen);
    }

    pub fn partial(&self, partial_index_type: char, alpha_type: char) -> Vec<InnerProduct> {
        let mut out = Vec::new();
        if let Some(bra_kdelta) = self.bra.partial(partial_index_type) {
            let mut inside = self.inner.clone();
            inside.push(Gen { alpha_type });
            let new_ip = InnerProduct::new(
                -1.0 * self.scalar,
                self.bra.clone(),
                inside,
                self.ket.clone(),
                Some(bra_kdelta),
            );
            out.push(new_ip);
        }
        if let Some(ket_kdelta) = self.ket.partial(partial_index_type) {
            let mut inside = self.inner.clone();
            inside.push(Gen { alpha_type });
            let new_ip = InnerProduct::new(
                self.scalar,
                self.bra.clone(),
                inside,
                self.ket.clone(),
                Some(ket_kdelta),
            );
            out.push(new_ip);
        }
        out
    }

    pub fn collapse_delta(&mut self, delta: &KDelta) {
        self.bra.collapse_delta(delta);
        self.ket.collapse_delta(delta);
        if Some(delta) == self.delta.as_ref() {
            self.delta = None
        }
    }

    pub fn collapse_alpha(&self, other: &InnerProduct, alpha: char) -> (Term, Term) {
        let mut ip1 = self.clone();
        let mut ip2 = other.clone();
        ip1.clear_inner_alpha(alpha);
        ip2.clear_inner_alpha(alpha);

        let new_ip1 = InnerProduct::new(
            ip1.get_scalar(),
            ip1.get_bra(),
            ip1.get_inner(),
            ip2.get_bra(),
            ip1.get_delta(),
        );
        let new_ip2 = InnerProduct::new(
            ip2.get_scalar(),
            ip1.get_ket(),
            ip2.get_inner(),
            ip2.get_ket(),
            ip2.get_delta(),
        );
        let new_ip3 = InnerProduct::new(
            -1.0 * ip1.get_scalar(),
            ip1.get_bra(),
            ip1.get_inner(),
            ip2.get_ket(),
            ip1.get_delta(),
        );
        let new_ip4 = InnerProduct::new(
            ip2.get_scalar(),
            ip1.get_ket(),
            ip2.get_inner(),
            ip2.get_bra(),
            ip2.get_delta(),
        );

        let t1 = Term::new(vec![new_ip1, new_ip2]);
        let t2 = Term::new(vec![new_ip3, new_ip4]);

        (t1, t2)
    }

    pub fn is_constant(&self) -> bool {
        self.bra == self.ket && self.delta == None && self.inner.is_empty() && self.scalar == 1.0
    }

    fn switch_order(&mut self) {
        let new_bra = self.get_ket();
        let new_ket = self.get_bra();
        self.bra = new_bra;
        self.ket = new_ket;
        if self.inner.len() == 1 {
            self.scalar *= -1.0;
        }
    }

    pub fn do_shift(&mut self, shift_amount: i8, index: usize) {
        self.bra.do_shift(shift_amount, index);
        self.ket.do_shift(shift_amount, index);
    }

    pub fn order_bra_kets(&mut self) {
        if self.bra > self.ket {
            self.switch_order();
        }
    }
}

impl PartialOrd for InnerProduct {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.bra.cmp(&other.bra) {
            Ordering::Equal => match self.ket.cmp(&other.ket) {
                Ordering::Equal => Some(Ordering::Equal),
                Ordering::Greater => Some(Ordering::Greater),
                Ordering::Less => Some(Ordering::Less),
            },
            Ordering::Greater => Some(Ordering::Greater),
            Ordering::Less => Some(Ordering::Less),
        }
    }
}

impl fmt::Display for InnerProduct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut my_str = "".to_string();
        if self.scalar != 1.0 {
            my_str += &self.scalar.to_string();
        }
        my_str += "<";
        my_str += &self.bra.get_string();
        my_str += "|";
        for g in &self.inner {
            my_str += "T";
            my_str += &g.alpha_type.to_string();
        }
        if !self.inner.is_empty() {
            my_str += "|";
        }
        my_str += &self.ket.get_string();
        my_str += ">";
        if let Some(delt) = &self.delta {
            my_str += &delt.to_string();
        }
        write!(f, "{}", my_str)
    }
}

impl MulAssign<f64> for InnerProduct {
    fn mul_assign(&mut self, rhs: f64) {
        self.scalar *= rhs;
    }
}

impl Mul<InnerProduct> for InnerProduct {
    type Output = Term;
    // This is overly simplified, once the derivatives are added this will change a lot!

    fn mul(self, rhs: Self) -> Term {
        Term::new(vec![self, rhs])
    }
}

impl Mul<Term> for InnerProduct {
    type Output = Term;

    fn mul(self, rhs: Term) -> Term {
        rhs * self // this just adds the InnerProduct to the term's ip: Vec<InnerProduct>
    }
}

impl Mul<f64> for InnerProduct {
    type Output = Self;
    // Returns a new InnerProduct with the scalar scaled by rhs

    fn mul(self, rhs: f64) -> InnerProduct {
        let mut dup = self;
        dup.scalar *= rhs;
        dup
    }
}

impl Mul<InnerProduct> for f64 {
    type Output = InnerProduct;

    // this implements left multiplication by scalar by calling the right multiplication
    fn mul(self, rhs: InnerProduct) -> InnerProduct {
        rhs * self
    }
}

impl Mul<Vec<InnerProduct>> for InnerProduct {
    type Output = Vec<Term>;

    fn mul(self, rhs: Vec<InnerProduct>) -> Vec<Term> {
        let mut out = Vec::new();
        for item in rhs {
            out.push(self.clone() * item);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::inner_product::*;

    #[test]
    fn collapse_alpha_1() {
        let ip1 = InnerProduct::new(
            1.0,
            BraKet::new('l', 'x', vec![1, 0, 0, 0]),
            vec![Gen::new('a')],
            BraKet::new('l', 'x', vec![0, 1, 0, 0]),
            None,
        );
        let ip2 = InnerProduct::new(
            1.0,
            BraKet::new('l', 'x', vec![0, 0, 1, 0]),
            vec![Gen::new('a')],
            BraKet::new('l', 'x', vec![0, 0, 0, 1]),
            None,
        );
        let (t1, t2) = ip1.collapse_alpha(&ip2, 'a');
        assert_eq!(
            t1,
            Term::new(vec![
                InnerProduct::new(
                    1.0,
                    BraKet::new('l', 'x', vec![1, 0, 0, 0]),
                    vec![],
                    BraKet::new('l', 'x', vec![0, 0, 1, 0]),
                    None
                ),
                InnerProduct::new(
                    1.0,
                    BraKet::new('l', 'x', vec![0, 1, 0, 0]),
                    vec![],
                    BraKet::new('l', 'x', vec![0, 0, 0, 1]),
                    None
                )
            ])
        );
        assert_eq!(
            t2,
            Term::new(vec![
                InnerProduct::new(
                    -1.0,
                    BraKet::new('l', 'x', vec![1, 0, 0, 0]),
                    vec![],
                    BraKet::new('l', 'x', vec![0, 0, 0, 1]),
                    None
                ),
                InnerProduct::new(
                    1.0,
                    BraKet::new('l', 'x', vec![0, 1, 0, 0]),
                    vec![],
                    BraKet::new('l', 'x', vec![0, 0, 1, 0]),
                    None
                )
            ])
        );
    }
}
