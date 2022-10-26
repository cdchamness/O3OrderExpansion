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
    let basic_ket = BraKet::new('l', 'y', vec![1,1]);
    
    let basic_ip = InnerProduct::new(3.0, basic_bra, Vec::new(), basic_ket, None);
    let mut basic_term = Term::new(vec![basic_ip.clone()]);
    basic_term.combine_scalars();
    println!("{}", basic_term);
    let d_basic_term = basic_term.partial('x', 'a');
    for test_term in d_basic_term {
    	println!("{}", test_term);
    }



    let test_bra = BraKet::new('l', 'y', vec![0,1]);
    let test_delta = KDelta::new('x', 'y', vec![0,1]);
    let new = test_bra.collapse_delta(&test_delta);
    println!("{}{} = {}", test_bra, test_delta, new);




    
}