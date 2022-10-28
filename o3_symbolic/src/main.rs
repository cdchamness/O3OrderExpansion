mod bra_ket;
mod inner_product;
mod term;
mod kdelta;

use crate::kdelta::*;
use crate::bra_ket::*;
use crate::inner_product::*;
use crate::term::*;


fn main() {
    let basic_bra = BraKet::new('l', 'y', vec![0,0]);
    let basic_ket = BraKet::new('l', 'y', vec![0,1]);
    
    let basic_bra2 = BraKet::new('l', 'y', vec![0,1]);
    let basic_ket2 = BraKet::new('l', 'y', vec![1,1]);
    
    let basic_ip = InnerProduct::new(3.0, basic_bra, Vec::new(), basic_ket, None);
    let basic_ip2 = InnerProduct::new(3.0, basic_bra2, Vec::new(), basic_ket2, None);
    let mut new_term = Term::new(vec![basic_ip, basic_ip2]);
    new_term.combine_scalars();
    println!("starting term: {}\n", new_term);

    let d_new_term = new_term.partial('x', 'a');
    for dt in d_new_term {
        let mut rdt = dt.reduce();
        println!("\nFirst derivative: {}", rdt);
        let dd_terms = rdt.partial('x', 'a');
        println!("\nAfter both derivatives");
        for mut ddt in dd_terms {
            ddt.combine_scalars();
            let result = ddt.reduce();
            println!("{}", result);
        }
    }
    


}