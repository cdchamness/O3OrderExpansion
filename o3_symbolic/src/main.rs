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
    for t in lapp {
        println!("{}", t);
    }
}
