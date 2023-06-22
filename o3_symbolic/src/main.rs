mod bra_ket;
mod inner_product;
mod kdelta;
mod term;
mod tools;

use crate::inner_product::*;
use crate::term::*;

use ordered_float::OrderedFloat;
extern crate nalgebra as na;

use std::collections::HashMap;

pub fn get_next_gp_from_prev_order(previous_order: Vec<Term>) -> Vec<Term> {
    let mut out = Vec::new();
    let mut prev_order = previous_order.clone();
    for term in &mut prev_order {
        term.extend_shift_len();
    }
    for term in prev_order {
        let len = term.get_shift_index_len();
        let new_term = OrderedFloat(0.5) * Term::new(vec![InnerProduct::basic(len)]);
        let mut gp = term.gradiant_product(new_term.clone(), 'x', 'a');
        for t in &mut gp {
            if let Some(_) = t.reduce() {
                t.add_term_to_vec(&mut out);
            }
        }
    }
    out
}

pub fn get_lap_terms_from_gp(
    grad_prod_result: Vec<Term>,
    lap_hash_map: &mut HashMap<Term, Vec<Term>>,
) -> Vec<Term> {
    let mut lap = Vec::new();
    for term in grad_prod_result {
        let mut tc = term.clone();
        tc.set_scalar(OrderedFloat(1.0));
        tc.set_index_type('y');
        let ddts = match lap_hash_map.get(&tc) {
            Some(laplacian) => laplacian.clone(),
            None => {
                let laplacian = tc.lapalacian('x', 'a');
                lap_hash_map.insert(tc.clone(), laplacian.clone());
                laplacian
            }
        };
        for mut ddt in ddts.clone() {
            if let Some(_) = ddt.reduce() {
                ddt.add_term_to_vec(&mut lap);
            }
        }
    }
    let closed_lap = get_closed_lap(lap.clone(), lap_hash_map);
    closed_lap
}

pub fn get_closed_lap(lap: Vec<Term>, lap_hash_map: &mut HashMap<Term, Vec<Term>>) -> Vec<Term> {
    let mut closed_lap = lap.clone();
    let mut i = 0;
    while i < closed_lap.len() {
        let mut tc = closed_lap[i].clone();
        tc.set_scalar(OrderedFloat(1.0));
        tc.set_index_type('y');
        let ddts = match lap_hash_map.get(&tc) {
            Some(laplacian) => laplacian.clone(),
            None => {
                let laplacian = tc.lapalacian('x', 'a');
                lap_hash_map.insert(tc.clone(), laplacian.clone());
                laplacian
            }
        };
        for mut ddt in ddts.clone() {
            if let Some(_) = ddt.reduce() {
                ddt.add_term_to_vec(&mut closed_lap);
            }
        }
        i += 1;
    }
    for t in closed_lap.iter_mut() {
        t.set_scalar(OrderedFloat(1.0));
        t.set_index_type('y');
    }
    closed_lap
}

pub fn build_matricies(
    gradiant_product: Vec<Term>,
    lap: Vec<Term>,
    lap_hash_map: &mut HashMap<Term, Vec<Term>>,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
    let dim = lap.len();
    let mut gp_vec = na::DVector::<f64>::zeros(dim);
    let mut i = 0;
    while i < lap.len() {
        for gpt in gradiant_product.clone() {
            let mut gptc = gpt.clone();
            let scalar = gptc.extract_scalar();
            gptc.set_index_type('y');
            if let Some(_) = gptc.clone() + lap[i].clone() {
                gp_vec[i] = scalar.0;
            }
        }
        i += 1;
    }

    let mut lap_mat = na::DMatrix::<f64>::zeros(dim, dim);
    let mut j = 0;
    while j < lap.len() {
        let ddts = match lap_hash_map.get(&lap[j]) {
            Some(v) => v.clone(),
            None => {
                let tc = &lap[j].clone();
                let laplacian = tc.lapalacian('x', 'a');
                lap_hash_map.insert(tc.clone(), laplacian.clone());
                laplacian
            }
        };
        let mut i = 0;
        while i < lap.len() {
            for ddt in &ddts {
                let mut ddtc = ddt.clone();
                let scalar = ddtc.extract_scalar();
                ddtc.set_index_type('y');
                if let Some(_) = ddtc + lap[i].clone() {
                    lap_mat[(i, j)] = scalar.0
                }
            }
            i += 1;
        }
        j += 1;
    }
    (gp_vec, lap_mat)
}

pub fn get_next_order(
    previous_order: Vec<Term>,
    lap_hash_map: &mut HashMap<Term, Vec<Term>>,
) -> Vec<Term> {
    let gp = get_next_gp_from_prev_order(previous_order);
    let lap = get_lap_terms_from_gp(gp.clone(), lap_hash_map);
    let (gp_vec, lap_mat) = build_matricies(gp, lap.clone(), lap_hash_map);
    println!("{}", gp_vec);
    println!("{}", lap_mat);
    let lap_inv = match lap_mat.try_inverse() {
        Some(inv) => inv,
        None => panic!("Could not invert Laplacian Matrix!"),
    };
    let result = lap_inv * gp_vec;

    let mut i = 0;
    let mut next_order = vec![];
    while i < result.len() {
        next_order.push(OrderedFloat(result[i]) * lap[i].clone());
        i += 1;
    }
    next_order
}

fn main() {
    let mut lap_hm = HashMap::new();
    let mut prev_order = vec![OrderedFloat(0.125) * Term::new(vec![InnerProduct::basic(1)])];
    println!("Order 0:");
    println!("{}", prev_order[0].clone());
    for i in 1..=3 {
        let next_order = get_next_order(prev_order.clone(), &mut lap_hm);
        println!("\nOrder {}:", i);
        for t in &next_order {
            println!("{}", t);
        }
        prev_order = next_order.clone();
    }

    //println!("\n\nLaplacian HashMap");
    //for (key, val) in lap_hm.iter() {
    //    let mut disp_string = format!("\n-∇^2 {} =", key);
    //    for t in val {
    //        disp_string += format!(" {} +", t).as_str();
    //    }
    //    disp_string.pop();
    //    println!("{}", disp_string);
    //}
}

// This workflow has been implemented in get_next_order()
// which is what main() now uses to loop to higher orders
// by plugging in the output back as input.
// The new function is much less verbose so I am keeping old_main()
// incase I want to use the previous made println!()'s.
/*
fn old_main() {
    let mut lap_hm = HashMap::new();
    // 0th Order Soln. As this is easy to calculate we will start from here
    let start_term = OrderedFloat(0.125) * Term::new(vec![InnerProduct::basic(1)]);
    println!("0th order:\n{}", start_term);
    let start_order = vec![start_term];
    let next_gp = get_next_gp_from_prev_order(start_order);
    println!("\n\nGP Result:");
    for term in &next_gp {
        println!("{}", term);
    }
    println!("\n\nLaplacian Result:");
    let lap = get_lap_terms_from_gp(next_gp.clone(), &mut lap_hm);
    for term in &lap {
        println!("{}", term);
    }
    println!("\n\nLaplacian HashMap");
    for (key, val) in lap_hm.iter() {
        let mut disp_string = format!("-∇^2 {} =", key);
        for t in val {
            disp_string += format!(" {} +", t).as_str();
        }
        disp_string.pop();
        println!("{}", disp_string);
    }
    let (gp_vec, lap_mat) = build_matricies(next_gp, lap.clone(), &mut lap_hm);
    println!("\nGP vector: {}", gp_vec);
    println!("Laplacian Matrix: {}", lap_mat);
    let lap_inv = match lap_mat.try_inverse() {
        Some(inv) => inv,
        None => panic!("Could not invert Laplacian Matrix!"),
    };
    println!("Laplacian Inverse: {}", lap_inv);
    let result = lap_inv * gp_vec;
    println!("\nResult: {}", result);

    let mut i = 0;
    let mut start_order = vec![];
    while i < result.len() {
        start_order.push(OrderedFloat(result[i]) * lap[i].clone());
        i += 1;
    }

    for j in &start_order {
        println!("{}", j);
    }
}
*/
