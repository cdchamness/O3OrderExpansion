use std::fmt;
use std::ops::Mul;

use crate::kdelta::*;
use crate::inner_product::*;

pub struct Term {
    ips: Vec<InnerProduct>
}

impl Term {

    pub fn new(ips: Vec<InnerProduct>) -> Term {
        Term { ips }
    }

    pub fn combine_scalars(&mut self) {
        let mut accum = 1.0;
        for ip in &mut self.ips {
            accum *= ip.extract_scalar()
        }
        self.ips[0].set_scalar(accum);
    }

    fn duplicate(&self) -> Term {
        let mut ips = Vec::new();
        for ip in &self.ips {
            ips.push(ip.clone())
        } 
        Term { ips }
    }

    pub fn partial(&mut self, partial_index_type: char, alpha_type: char) -> Vec<Term> {
        let mut out = Vec::new();
        let ip_count = self.ips.len();
        for _ in 0..ip_count {
            let ip = self.ips.pop().unwrap();
            let d_ips = ip.partial(partial_index_type, alpha_type);
            for d_ip in d_ips {
                out.push(d_ip * self.duplicate());
            }
            self.ips.insert(0,ip);
        }
        
        out
    }

    pub fn collapse_delta(&self, delta: &KDelta) -> Term {
        let mut new_ips = Vec::new();
        for ip in &self.ips {
            new_ips.push(ip.collapse_delta(delta));
        }
        Term { ips: new_ips }
    }


    pub fn reduce(&self) -> Term {
        let mut new_term = Term { ips: self.ips.clone() };
        while let Some(next_delta) = new_term.get_delta_index() {
            new_term = self.collapse_delta(&next_delta);
        }
        new_term 
    }

    fn get_delta_index(&self) -> Option<KDelta> {
        for ip in &self.ips {
            if let Some(delta) = ip.delta.clone() {
                return Some(delta);
            }
        }
        None
    }
}


impl Mul<InnerProduct> for Term {
    type Output = Term;

    fn mul(self, rhs: InnerProduct) -> Term {
        let mut ips = self.ips;
        ips.push(rhs);
        Term::new(ips)
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut disp_string = "".to_string();
        for ip in &self.ips {
            disp_string += &ip.to_string();
        }

        write!(f, "{}", disp_string)
    }
}