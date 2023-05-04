mod bra_ket;
mod inner_product;
mod kdelta;
mod term;

use ordered_float::OrderedFloat;

use crate::bra_ket::*;
use crate::inner_product::*;
use crate::term::*;

use std::collections::HashMap;

pub fn get_next_gp_from_prev_order(previous_order: Vec<Term>) -> Vec<Term> {
    let mut out = Vec::new();
    let mut prev_order = previous_order.clone();
    for term in &mut prev_order {
        term.extend_shift_len();
    }
    let len = prev_order[0].get_shift_index_len();
    let new_term = Term::new(vec![InnerProduct::basic(len)]);
    for term in prev_order {
        let mut gp = term.gradiant_product(new_term.clone(), 'y', 'a');
        for t in &mut gp {
            if let Some(_) = t.reduce() {
                t.add_term_to_vec(&mut out);
            }
        }
    }
    out
}

pub fn get_lapp_terms_from_gp(
    grad_prod_result: Vec<Term>,
    lapp_hash_map: &mut HashMap<Term, Vec<Term>>,
) -> (Vec<Term>, &mut HashMap<Term, Vec<Term>>) {
    let mut lapp = Vec::new();
    for term in grad_prod_result {
        let mut tc = term.clone();
        tc.set_lattice_type('y');
        let mut ddts = tc.lappalacian('x', 'a');
        lapp_hash_map.insert(tc, ddts.clone());
        for ddt in &mut ddts {
            if let Some(_) = ddt.reduce() {
                ddt.add_term_to_vec(&mut lapp)
            }
        }
    }
    for term in &mut lapp {
        term.set_scalar(OrderedFloat(1.0))
    }
    (lapp, lapp_hash_map)
}

pub fn find_order_s(lapp_terms: Vec<Term>, grad_prod_result: Vec<Term>) -> Vec<Term> {
    let mut out = Vec::new();
    for l_term in lapp_terms {
        let mut ltc = l_term.clone();
        ltc.set_lattice_type('y');
        let mut ddts = ltc.lappalacian('x', 'a');
        for ddt in &mut ddts {
            if let Some(_) = ddt.reduce() {}
        }
    }
    out
}

pub fn get_next_order(previous_order: Vec<Term>) -> Vec<Term> {
    let gp = get_next_gp_from_prev_order(previous_order);
    let (lapp, _) = get_lapp_terms_from_gp(gp, &mut HashMap::new());
    lapp
}

fn main() {
    let mut lapp_hash_map = HashMap::new();
    let start_term = 0.5 * Term::new(vec![InnerProduct::basic(1)]);
    println!("0th order:\n{}", start_term);
    let start_order = vec![start_term];
    let next_gp = get_next_gp_from_prev_order(start_order);
    println!("\n\nGP Result:");
    for term in &next_gp {
        println!("{}", term);
    }
    println!("\n\nLapp Result:");
    let (lapp, hm) = get_lapp_terms_from_gp(next_gp, &mut lapp_hash_map);
    for term in &lapp {
        println!("{}", term);
    }
    println!("{:?}", hm)
}

fn main2() {
    let basic_bra = BraKet::new('l', 'y', vec![0]);
    let basic_ket = BraKet::new('l', 'y', vec![1]);

    let basic_ip = InnerProduct::new(OrderedFloat(1.0), basic_bra, Vec::new(), basic_ket, None);

    let mut new_term = Term::new(vec![basic_ip]);
    let lapp = new_term.lappalacian('x', 'a');
    for mut t in lapp {
        t.sort_ips();
        println!("{}", t);
    }

    println!("\nDoing Gradiant Product Test\n");

    let bk1 = BraKet::new('l', 'y', vec![0, 0]);
    let bk2 = BraKet::new('l', 'y', vec![1, 0]);
    let bk3 = BraKet::new('l', 'y', vec![0, 1]);

    let ip1 = InnerProduct::new(OrderedFloat(1.0), bk1.clone(), vec![], bk2.clone(), None);
    let ip2 = InnerProduct::new(OrderedFloat(1.0), bk1.clone(), vec![], bk3.clone(), None);

    let t1 = Term::new(vec![ip1.clone()]);
    let t2 = Term::new(vec![ip2.clone()]);

    for t in t1.gradiant_product(t2, 'x', 'a') {
        println!("{}", t);
    }

    println!("\n\nDoing parity_transform Test:\n");
    let mut t3 = Term::new(vec![ip1, ip2]);
    t3.reduce();
    println!("{}", t3);
    t3.parity_transform(0);
    t3.shift_down();
    t3.sort_ips();
    println!("{}", t3);
    t3.parity_transform(1);
    t3.shift_down();
    t3.sort_ips();
    println!("{}", t3);
    t3.parity_transform(0);
    t3.shift_down();
    t3.sort_ips();
    println!("{}", t3);
    t3.parity_transform(1);
    t3.shift_down();
    t3.sort_ips();
    println!("{}", t3);
    t3.parity_reduce(vec![true, true]);
    println!("{}", t3);

    println!("\nDoing shift_down test: \n");

    let bk4 = BraKet::new('l', 'y', vec![0, -1]);
    let bk5 = BraKet::new('l', 'y', vec![-1, -2]);
    let ip3 = InnerProduct::new(OrderedFloat(1.0), bk4, vec![], bk5, None);
    let mut t3 = Term::new(vec![ip3]);
    println!("{}", t3);
    t3.shift_down();
    println!("{}", t3);
    if let Some(mut new_ip) = t3.get_ips().pop() {
        new_ip.order_bra_kets();
        println!("{}", new_ip);
    }
}
