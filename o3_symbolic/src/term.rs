use ordered_float::OrderedFloat;
use std::fmt;
use std::ops::{Add, Mul};

use crate::inner_product::*;
use crate::kdelta::*;

#[derive(Clone, Debug, Eq, Hash)]
pub struct Term {
    ips: Vec<InnerProduct>,
}

impl Term {
    pub fn new(ips: Vec<InnerProduct>) -> Term {
        Term { ips }
    }

    pub fn extract_scalar(&mut self) -> OrderedFloat<f64> {
        self.ips[0].extract_scalar()
    }

    pub fn get_scalar_val(&self) -> OrderedFloat<f64> {
        self.ips[0].get_scalar()
    }

    pub fn set_scalar(&mut self, value: OrderedFloat<f64>) {
        self.scalar_reduce();
        self.ips[0].set_scalar(value);
    }

    pub fn scalar_reduce(&mut self) {
        let mut accum = OrderedFloat(1.0);
        for ip in &mut self.ips {
            accum *= ip.extract_scalar(); //returns scalar for ip and sets its own scalar to 1.0
        }
        self.ips[0] *= accum
    }

    pub fn get_ips(&self) -> Vec<InnerProduct> {
        self.ips.clone()
    }

    pub fn get_shift_index_len(&self) -> usize {
        self.ips[0].get_bra().get_shift().len()
    }

    pub fn extend_shift_len(&mut self) {
        for ip in &mut self.ips {
            ip.extend_shift_len();
        }
    }

    pub fn set_lattice_type(&mut self, new_lattice_type: char) {
        for ip in &mut self.ips {
            ip.set_lattice_type(new_lattice_type);
        }
    }

    pub fn duplicate(&self) -> Term {
        let mut ips = Vec::new();
        for ip in &self.ips {
            ips.push(ip.clone())
        }
        Term::new(ips)
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
            self.ips.insert(0, ip);
        }

        out
    }

    pub fn lappalacian(&mut self, partial_index_type: char, alpha_type: char) -> Vec<Term> {
        let mut lapp = Vec::new();
        let d_terms = self.partial(partial_index_type, alpha_type);
        for mut dt in d_terms {
            dt.collapse_all_deltas();
            let dd_terms = dt.partial(partial_index_type, alpha_type);
            for mut ddt in dd_terms {
                ddt.scalar_reduce();
                ddt.collapse_all_deltas();
                let out = ddt.alpha_reduce(alpha_type);
                for mut term in out {
                    if let Some(_) = term.reduce() {
                        term.add_term_to_vec(&mut lapp);
                    }
                }
            }
        }
        lapp
    }

    pub fn gradiant_product(
        &self,
        other: Term,
        partial_index_type: char,
        alpha_type: char,
    ) -> Vec<Term> {
        let mut gp = Vec::new();
        let mut t1 = self.clone();
        let mut t2 = other;
        for mut d_t1 in t1.partial(partial_index_type, alpha_type) {
            d_t1.collapse_all_deltas();
            for mut d_t2 in t2.partial(partial_index_type, alpha_type) {
                d_t2.collapse_all_deltas();
                let mut out = d_t1.clone() * d_t2;
                for mut term in out.alpha_reduce(alpha_type) {
                    if let Some(_) = term.reduce() {
                        term.add_term_to_vec(&mut gp);
                    }
                }
            }
        }
        gp
    }

    pub fn add_term_to_vec(&self, v: &mut Vec<Term>) {
        // updates v to include term
        let mut counter = 0;
        while counter < v.len() {
            let mut current_parity = vec![false; self.get_shift_index_len()];
            if let Some(t) = v.pop() {
                // grabs last element out of v
                loop {
                    // start parity loop

                    // make term have same parity as 'current_partiy'
                    let mut term_cl = self.clone();
                    term_cl.parity_reduce(current_parity.clone());

                    // Compares if terms are identicial up to a scalar
                    if let Some(res) = t.clone() + self.clone() {
                        // if it is, add their sum to v, exit the function
                        v.push(res);
                        return;
                    } else {
                        // if it isn't, update pairty type and continue loop
                        current_parity = Self::get_next_parity(current_parity);
                    }

                    // if parity has completely cycled, this term's class isnt in v
                    // => we update the counter, return t into v at front, and break parity loop
                    if current_parity == vec![false; self.get_shift_index_len()] {
                        counter += 1;
                        v.insert(0, t);
                        break;
                    }
                }
            }
        }
        // if we got through parity loop for every term in v, then the new term is different than all other elements => add new term
        v.push(self.clone());
    }

    pub fn collapse_delta(&mut self, delta: &KDelta) {
        for ip in &mut self.ips {
            ip.collapse_delta(delta);
        }
    }

    pub fn collapse_all_deltas(&mut self) {
        while let Some(next_delta) = self.get_next_delta() {
            self.collapse_delta(&next_delta);
        }
    }

    fn get_next_delta(&self) -> Option<KDelta> {
        for ip in &self.ips {
            if let Some(delta) = ip.get_delta() {
                return Some(delta);
            };
        }
        None
    }

    fn get_alpha_count(&self, alpha: char) -> Vec<usize> {
        let mut count: Vec<usize> = Vec::new();
        for ip in &self.ips {
            let mut counts = 0;
            for gen in ip.get_inner() {
                if gen.get_type() == alpha {
                    counts += 1
                }
            }
            count.push(counts);
        }
        count
    }

    pub fn alpha_reduce(&mut self, alpha: char) -> Vec<Term> {
        let inner_count = self.get_alpha_count(alpha);
        if inner_count.iter().sum::<usize>() == 2 {
            // if there are 2 inners, either they are on the same InnerProduct or they are split
            if let Some(max_value) = inner_count.iter().max() {
                if *max_value == 2 {
                    // They are both on the same InnerProduct
                    let index = inner_count.iter().position(|&x| x == 2).unwrap();
                    self.ips[index].clear_inner_alpha(alpha);
                    self.ips[index] *= OrderedFloat(-2.0);

                    // duplicate self, change ips at index to new_ip
                    vec![self.duplicate()]
                } else {
                    // They are split across multiple InnerProducts
                    let mut alpha_indices = Vec::new();
                    let mut other_indices = Vec::new();
                    for (i, x) in inner_count.iter().enumerate() {
                        if *x == 1 {
                            alpha_indices.push(i);
                        } else {
                            other_indices.push(i);
                        }
                    }

                    let mut other_ips = Vec::new();
                    for index in other_indices {
                        other_ips.push(self.ips[index].clone())
                    }
                    let other_term = Term { ips: other_ips };

                    // indices should have 2
                    let ip1 = &self.ips[alpha_indices[0]];
                    let ip2 = &self.ips[alpha_indices[1]];

                    let (term1, term2) = ip1.collapse_alpha(ip2, alpha);
                    vec![other_term.duplicate() * term1, other_term * term2]
                }
            } else {
                // Should not be possible to ever reach
                unreachable!("max is only called if the sum == 2 => there must be a max");
            }
        } else {
            // if there are not 2 'inners' of the same type this does nothing.
            vec![self.duplicate()]
        }
    }

    pub fn reduce(&mut self) -> Option<()> {
        if let Some(_) = self.remove_constants() {
            self.shift_down();
            self.sort_ips();
            self.scalar_reduce();
            Some(())
        } else {
            None
        }
    }

    pub fn remove_constants(&mut self) -> Option<()> {
        let mut accum = OrderedFloat(1.0);
        let mut out = Vec::new();
        for ip in &self.ips {
            if !ip.is_constant() {
                out.push(ip.clone());
            } else {
                accum *= ip.get_scalar();
            }
        }
        if !out.is_empty() {
            out[0] *= accum;
        } else {
            return None;
        }
        self.ips = out;
        Some(())
    }

    pub fn shift_down(&mut self) {
        let ips_len = self.get_ips().len();
        for i in 0..ips_len {
            let shifts_to_do = self.ips[i].get_bra().get_neg_shifts();
            for (shift, index) in shifts_to_do {
                self.do_shift(shift, index);
            }
            let shifts_to_do = self.ips[i].get_ket().get_neg_shifts();
            for (shift, index) in shifts_to_do {
                self.do_shift(shift, index);
            }
        }
    }

    fn do_shift(&mut self, shift_amount: i8, index: usize) {
        let mut new_ips = Vec::new();
        let mut current_ips = self.get_ips();
        for ip in current_ips.iter_mut() {
            ip.do_shift(shift_amount, index);
            new_ips.push(ip.clone());
        }
        self.ips = new_ips;
    }

    pub fn get_total_shifts_by_index(&self) -> Vec<i8> {
        let mut v: Vec<i8> = vec![0; self.get_shift_index_len()];
        for ip in &self.ips {
            let bra_shift = ip.get_bra().get_shift();
            for (i, b) in bra_shift.iter().enumerate() {
                v[i] += b;
            }
            let ket_shift = ip.get_ket().get_shift();
            for (i, k) in ket_shift.iter().enumerate() {
                v[i] += k;
            }
        }
        v
    }

    pub fn sort_ips(&mut self) {
        let mut new_ips = Vec::new();
        for mut ip in self.get_ips() {
            ip.order_bra_kets();
            new_ips.push(ip);
        }
        new_ips.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.ips = new_ips;
    }

    pub fn parity_reduce(&mut self, parities: Vec<bool>) {
        for (index, val) in parities.iter().enumerate() {
            if *val {
                self.parity_transform(index);
            }
        }
        self.reduce();
    }

    pub fn parity_transform(&mut self, index: usize) {
        for ip in self.ips.iter_mut() {
            ip.parity_transform(index);
        }
    }

    pub fn get_next_parity(current_parity: Vec<bool>) -> Vec<bool> {
        let mut v = current_parity.clone();
        for (i, p) in current_parity.iter().enumerate() {
            if *p {
                v[i] = false;
            } else {
                v[i] = true;
                return v;
            }
        }
        v
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

impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        let my_len = self.ips.len();
        let other_len = other.ips.len();
        if my_len == other_len {
            let mut counter = 0;
            for i in 0..my_len {
                if self.ips[i] == other.ips[i] {
                    counter += 1;
                }
            }
            counter == my_len // only true if ips are exactly the same (except scalar)
        } else {
            false
        }
    }
}

impl Mul<OrderedFloat<f64>> for Term {
    type Output = Term;

    fn mul(self, rhs: OrderedFloat<f64>) -> Term {
        let mut ips = self.ips;
        let mut front = ips.remove(0);
        front *= rhs;
        ips.insert(0, front);
        Term::new(ips)
    }
}

impl Mul<Term> for OrderedFloat<f64> {
    type Output = Term;

    fn mul(self, rhs: Term) -> Term {
        rhs * self
    }
}

impl Mul<Term> for Term {
    type Output = Term;

    fn mul(self, rhs: Term) -> Term {
        let mut ips = self.ips;
        for new_ip in rhs.get_ips() {
            ips.push(new_ip);
        }
        Term::new(ips)
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut disp_string = self.get_scalar_val().clone().to_string();
        for ip in &self.ips {
            disp_string += &ip.to_string();
        }

        write!(f, "{}", disp_string)
    }
}

impl Add<Term> for Term {
    type Output = Option<Term>;

    fn add(self, rhs: Term) -> Option<Term> {
        // clone self and rhs and set clones scalars to 1.0
        let mut cl = self.clone();
        let mut rhs_cl = rhs.clone();
        let self_scalar = cl.extract_scalar();
        let rhs_scalar = rhs_cl.extract_scalar();

        // both scalars should be set to 1.0 for the clones => the comparison should only consider the structure as the scalar components will match
        if cl == rhs_cl {
            cl.set_scalar(self_scalar + rhs_scalar);
            if let Some(_) = cl.reduce() {
                Some(cl)
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{bra_ket::BraKet, inner_product::InnerProduct, term::Term};
    use ordered_float::OrderedFloat;

    #[test]
    fn next_parity1() {
        let v = vec![false, false, false];
        let p = Term::get_next_parity(v);
        assert_eq!(p, vec![true, false, false])
    }

    #[test]
    fn next_parity2() {
        let v = vec![true, true, false];
        let p = Term::get_next_parity(v);
        assert_eq!(p, vec![false, false, true])
    }

    #[test]
    fn next_parity3() {
        let v = vec![false, true, false];
        let p = Term::get_next_parity(v);
        assert_eq!(p, vec![true, true, false])
    }

    #[test]
    fn test_extend_shift() {
        let mut t = Term::new(vec![InnerProduct::new(
            OrderedFloat(1.0),
            BraKet::new('l', 'x', vec![0]),
            vec![],
            BraKet::new('l', 'x', vec![1]),
            None,
        )]);
        t.extend_shift_len();
        assert_eq!(
            t,
            Term::new(vec![InnerProduct::new(
                OrderedFloat(1.0),
                BraKet::new('l', 'x', vec![0, 0]),
                vec![],
                BraKet::new('l', 'x', vec![1, 0]),
                None
            )])
        );
    }

    #[test]
    fn test_extend_shift2() {
        let mut t = Term::new(vec![
            InnerProduct::new(
                OrderedFloat(1.0),
                BraKet::new('l', 'x', vec![0]),
                vec![],
                BraKet::new('l', 'x', vec![1]),
                None,
            ),
            InnerProduct::new(
                OrderedFloat(1.0),
                BraKet::new('l', 'x', vec![1]),
                vec![],
                BraKet::new('l', 'x', vec![2]),
                None,
            ),
        ]);
        t.extend_shift_len();
        assert_eq!(
            t,
            Term::new(vec![
                InnerProduct::new(
                    OrderedFloat(1.0),
                    BraKet::new('l', 'x', vec![0, 0]),
                    vec![],
                    BraKet::new('l', 'x', vec![1, 0]),
                    None
                ),
                InnerProduct::new(
                    OrderedFloat(1.0),
                    BraKet::new('l', 'x', vec![1, 0]),
                    vec![],
                    BraKet::new('l', 'x', vec![2, 0]),
                    None,
                )
            ])
        );
    }
}
