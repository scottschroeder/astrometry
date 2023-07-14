pub(crate) fn node_level(nodeid: usize) -> u32 {
    leftmost_significant_bit(nodeid)
}

pub(crate) fn kdtree_nnodes_to_nlevels(nnodes: usize) -> u32 {
    leftmost_significant_bit(nnodes + 1)
}

pub(crate) fn compute_levels(data_len: usize, leaf_size: usize) -> u32 {
    assert!(leaf_size > 0, "leaf size must be greater than zero");
    let nnodes = data_len / leaf_size;
    usize::BITS + 1 - nnodes.leading_zeros()
}

/// How far left is the most significant bit?
/// 0b0001 would be the 0-bit
/// 0b0011 would be the 1-bit
/// 0b1010 would be the 3-bit
#[inline]
fn leftmost_significant_bit(x: usize) -> u32 {
    // an_flsB
    assert!(
        x > 0,
        "x must be greater than 0 so that there is at least one flipped bit"
    );
    usize::BITS - 1 - x.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iterative(mut x: usize) -> u32 {
        assert!(x > 0);
        let mut bit = 0;
        while x != 1 {
            x >>= 1;
            bit += 1;
        }
        bit
    }

    fn verify_lsb(x: usize) -> u32 {
        let iter_res = iterative(x);
        let direct_res = leftmost_significant_bit(x);
        assert_eq!(
            direct_res, iter_res,
            "direct approach does not equal iterative"
        );
        direct_res
    }

    #[test]
    #[should_panic]
    fn check_zero_fails() {
        verify_lsb(0);
    }
    #[test]
    fn zeroth_bit() {
        assert_eq!(verify_lsb(0b0001), 0);
    }
    #[test]
    fn first_bit() {
        assert_eq!(verify_lsb(0b0010), 1);
        assert_eq!(verify_lsb(0b0011), 1);
    }
    #[test]
    fn u32_bits() {
        assert_eq!(verify_lsb(u32::MAX as usize), 31);
    }
}
