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
    let new_term = OrderedFloat(0.5) * Term::new(vec![InnerProduct::basic(len)]);
    for term in prev_order {
        let mut gp = term.gradiant_product(new_term.clone(), 'y', 'a');
        for t in &mut gp {
            if let Some(_) = t.reduce() {
                // two issues it does not fully reduce
                // 1. Does not kill constant InnerProducts i.e. <y_0|y_0>
                // 2. Does not fully account for different versions of the same Terms
                //    i.e. <y_0,0|y_1,0><y_1,0|y_1,1> == <y_1,0|y_0,0><y_0,0|y_0,1>
                t.add_term_to_vec(&mut out);
            }
        }
    }
    out
}

pub fn get_lapp_terms_from_gp(
    grad_prod_result: Vec<Term>,
    lapp_hash_map: &mut HashMap<Term, Vec<Term>>,
) -> Vec<Term> {
    let mut lapp = Vec::new();
    for term in grad_prod_result {
        let mut tc = term.clone();
        let scalar = tc.extract_scalar();
        tc.set_lattice_type('y');
        let ddts = tc.lappalacian('x', 'a');
        if !lapp_hash_map.contains_key(&tc) {
            lapp_hash_map.insert(tc, ddts.clone());
        }
        for ddt in ddts {
            let mut scaled_ddt = ddt * scalar;
            if let Some(_) = scaled_ddt.reduce() {
                println!("\nStarting Lapp:");
                for l in &lapp {
                    println!("{}", l);
                }
                println!("Term to add: {}", scaled_ddt);
                scaled_ddt.add_term_to_vec(&mut lapp);

                println!("Ending Lapp:");
                for l in &lapp {
                    println!("{}", l);
                }
            }
        }
    }

    println!("\n\n\n\n");
    lapp
}

/*
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
*/

pub fn get_next_order(previous_order: Vec<Term>) -> Vec<Term> {
    let gp = get_next_gp_from_prev_order(previous_order);
    let lapp = get_lapp_terms_from_gp(gp, &mut HashMap::new());
    lapp
}

fn main() {
    let mut lapp_hm = HashMap::new();
    // 0th Order Soln. As this is easy to calculate we will start from here
    let start_term = OrderedFloat(0.125) * Term::new(vec![InnerProduct::basic(1)]);
    println!("0th order:\n{}", start_term);
    let start_order = vec![start_term];
    let lapp = get_lapp_terms_from_gp(start_order, &mut lapp_hm);

    for term in &lapp {
        println!("{}", term);
    }
    let next_gp = get_next_gp_from_prev_order(lapp);
    println!("\n\nGP Result:");
    for term in &next_gp {
        println!("{}", term);
    }
    println!("\n\nLapp Result:");
    let lapp = get_lapp_terms_from_gp(next_gp, &mut lapp_hm);
    for term in &lapp {
        println!("{}", term);
    }
    for (key, val) in lapp_hm.iter() {
        println!("\nkey: {key}");
        let mut disp_string = "".to_owned();
        for t in val {
            disp_string += format!(" {} +", t).as_str();
        }
        disp_string.pop();
        println!("{}", disp_string);
    }
}
