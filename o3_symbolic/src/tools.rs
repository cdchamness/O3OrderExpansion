pub fn gen_all_permuations(n: usize) -> Vec<Vec<usize>> {
    let mut base: Vec<usize> = (0..n).collect();
    let mut perms = Vec::new();
    gen_perms(n, &mut base, &mut perms);
    perms
}

fn gen_perms(n: usize, base: &mut Vec<usize>, perms: &mut Vec<Vec<usize>>) {
    if n == 1 {
        perms.push(base.clone());
        return;
    }

    for i in 0..n {
        base.swap(i, n - 1);
        gen_perms(n - 1, base, perms);
        base.swap(i, n - 1);
    }
}

pub fn gen_pairs<T: Clone + PartialEq + Ord>(base: &mut [T]) -> Vec<Vec<(T, T)>> {
    // generates all unique sets of pairings for a given number
    // get the length of the base
    let n = base.len();
    base.sort(); // needed for checking for duplicate pairs or base needs to be presorted

    if n % 2 != 0 {
        // need an even number of items to generate pairings
        return Vec::new();
    }
    if n == 2 {
        // only 1 possible pairing so just return that as long as they are not the same element
        if base[0] == base[1] {
            return Vec::new();
        }
        return vec![vec![(base[0].clone(), base[1].clone())]];
    }

    // Recurive implementation
    let i = base[0].clone();
    let mut pairings: Vec<Vec<(T, T)>> = Vec::new();

    for (ind, j) in base.iter().enumerate() {
        // dont pair element up with another element with the same value
        if ind != 0 && j != &i {
            // generate a single pair
            let pair = (i.clone(), j.clone());

            // call 'gen_pairs' on the complement
            let mut complement: Vec<_> = [&base[1..ind], &base[ind + 1..]].concat();
            let complement_pairs = gen_pairs(&mut complement);

            // At this point 'complement_pairs' has all pairings of the unpaired items
            // we just need to add our 'pair' to that each pairing in complemnet pairs
            for mut pairing_set in complement_pairs {
                // pairing_set has type Vec<(T, T)> so we just add 'pair'
                pairing_set.push(pair.clone());
                pairing_set.sort(); // needed to check for duplicates
                if !pairings.contains(&pairing_set) {
                    pairings.push(pairing_set.clone());
                }
            }
        }
    }
    pairings
}
