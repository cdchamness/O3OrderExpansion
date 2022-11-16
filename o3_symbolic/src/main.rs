mod bra_ket;
mod inner_product;
mod kdelta;
mod term;

use crate::bra_ket::*;
use crate::inner_product::*;
use crate::term::*;

fn main() {
    let basic_bra = BraKet::new('l', 'y', vec![0, 0]);
    let basic_ket = BraKet::new('l', 'y', vec![0, 1]);

    let basic_bra2 = BraKet::new('l', 'y', vec![0, 1]);
    let basic_ket2 = BraKet::new('l', 'y', vec![1, 1]);

    let basic_ip = InnerProduct::new(3.0, basic_bra, Vec::new(), basic_ket, None);
    let basic_ip2 = InnerProduct::new(3.0, basic_bra2, Vec::new(), basic_ket2, None);
    let mut new_term = Term::new(vec![basic_ip.clone(), basic_ip2.clone()]);
    new_term.scalar_reduce();
    println!("Starting term: {}\n", new_term);

    let d_new_term = new_term.partial('x', 'a');
    for mut dt in d_new_term {
        dt.collapse_all_deltas();
        println!("\n\nFirst derivative: {}", dt);
        let dd_terms = dt.partial('x', 'a');
        println!("After both derivatives");
        for mut ddt in dd_terms {
            ddt.scalar_reduce();
            ddt.collapse_all_deltas();
            let terms = ddt.alpha_reduce('a');
            for mut t in terms {
                t.scalar_reduce();
                println!("{}", t);
            }
        }
    }

    println!("First implementation of Lapp\n\n\n\n\nNow time for the second: \n");

    let mut new_term = Term::new(vec![basic_ip, basic_ip2]);
    let lapp = new_term.lapalacian('x', 'a');
    for mut t in lapp {
        t.sort_ips();
        println!("{}", t);
    }

    println!("\n\n\n\n\nDoing Gradiant Product Test\n");

    let bk1 = BraKet::new('l', 'y', vec![0, 0]);
    let bk2 = BraKet::new('l', 'y', vec![1, 0]);
    let bk3 = BraKet::new('l', 'y', vec![0, 1]);

    let ip1 = InnerProduct::new(1.0, bk1.clone(), vec![], bk2.clone(), None);
    let ip2 = InnerProduct::new(1.0, bk1.clone(), vec![], bk3.clone(), None);

    let t1 = Term::new(vec![ip1]);
    let t2 = Term::new(vec![ip2]);

    for mut t in t1.gradiant_product(t2, 'x', 'a') {
        t.sort_ips();
        println!("{}", t);
    }

    println!("\n\n\n\n\n Doing shift_down test: \n");

    let bk4 = BraKet::new('l', 'y', vec![0, -1]);
    let bk5 = BraKet::new('l', 'y', vec![-1, -2]);
    let ip3 = InnerProduct::new(1.0, bk4, vec![], bk5, None);
    let mut t3 = Term::new(vec![ip3]);
    println!("{}", t3);
    t3.shift_down();
    println!("{}", t3);
    if let Some(mut new_ip) = t3.get_ips().pop() {
        new_ip.order_bra_kets();
        println!("{}", new_ip);
    }
}
