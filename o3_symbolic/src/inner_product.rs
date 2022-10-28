use std::ops::Mul;
use std::fmt;

use crate::bra_ket::*;
use crate::kdelta::*;
use crate::term::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Gen {
    alpha_type: char,
}

impl Gen {
    pub fn new(alpha_type: char) -> Gen {
        Gen { alpha_type }
    }
}



#[derive(Clone, Debug, PartialEq)]
pub struct InnerProduct {
    scalar: f64,
    bra: BraKet,
    inner: Vec<Gen>,
    ket: BraKet,
    pub delta: Option<KDelta>,
}

impl InnerProduct {

    pub fn new(scalar: f64, bra: BraKet, inner: Vec<Gen>, ket: BraKet, delta: Option<KDelta>) -> InnerProduct {
        InnerProduct {
            scalar,
            bra,
            inner,
            ket,
            delta,
        }
    }


    pub fn extract_scalar(&mut self) -> f64 {
        let num = self.scalar;
        self.scalar = 1.0;
        num
    }

    pub fn set_scalar(&mut self, num: f64) {
        self.scalar = num;
    }

    pub fn partial(&self, partial_index_type: char, alpha_type: char) -> Vec<InnerProduct> {
        let mut out = Vec::new();
        if let Some(bra_kdelta) = self.bra.partial(partial_index_type) {
            let mut inside = self.inner.clone();
            inside.push(Gen{alpha_type});
            let new_ip = InnerProduct::new(-1.0*self.scalar, self.bra.clone(), inside, self.ket.clone(), Some(bra_kdelta));
            out.push(new_ip);
        }
        if let Some(ket_kdelta) = self.ket.partial(partial_index_type) {
            let mut inside = self.inner.clone();
            inside.push(Gen{alpha_type});
            let new_ip = InnerProduct::new(self.scalar, self.bra.clone(), inside, self.ket.clone(), Some(ket_kdelta));
            out.push(new_ip);
        }
        out
    }

    pub fn collapse_delta(&self, delta: &KDelta) -> InnerProduct {
        let delta_clone = self.delta.clone();
        let new_bra = self.bra.collapse_delta(delta);
        let new_ket = self.ket.collapse_delta(delta);
        if delta_clone == self.delta {
            InnerProduct {
                scalar: self.scalar,
                bra: new_bra,
                inner: self.inner.clone(),
                ket: new_ket,
                delta: None,
            }
        }
        else {
            InnerProduct {
                scalar: self.scalar,
                bra: new_bra,
                inner: self.inner.clone(),
                ket: new_ket,
                delta: delta_clone,
            }
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











