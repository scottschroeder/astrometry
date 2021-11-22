use super::error::Error;
use super::kd_types::KdtMetadata;
use crate::kdtree::bits::compute_levels;
use crate::kdtree::bits::node_level;

mod rangesearch;

pub(crate) struct KDTree<'a, T, D> {
    pub(crate) metadata: KdtMetadata,
    /// Points owned by leaf nodes
    pub(crate) lr: &'a [u32],
    /// Permutation index
    pub(crate) perm: &'a [u32],
    pub(crate) cut: TreeCut<'a, T>,
    pub(crate) data: &'a [D],
    pub(crate) splitdim: &'a [u8],

    pub(crate) minval: Option<f64>,
    pub(crate) maxval: Option<f64>,

    /// kdtype per real
    pub(crate) scale: f64,
    /// real per kdtype
    pub(crate) invscale: f64,
}

impl<'a, T, D> KDTree<'a, T, D> {
    /// Number of data points in kdtree
    pub fn len(&self) -> usize {
        self.data.len()
    }

    fn bb_any(&self) -> bool {
        match self.cut {
            TreeCut::BoundingBox(b) => !b.is_empty(),
            TreeCut::SplitDim { .. } => false,
        }
    }

    // TODO I think we need to differentiate an owned vs borrowed KDTree
    pub fn new(ndata: usize, ndim: usize, leaf_size: usize) -> KDTree<'a, T, D> {
        let maxlevel = compute_levels(ndata, leaf_size);
        let nnodes = (1usize << maxlevel) - 1;
        let nbottom = 1usize << (maxlevel - 1);
        let metadata = KdtMetadata {
            kdt_type: todo!(),
            has_linear_lr: todo!("has linear lr for kdtree_new?"),
            ndata,
            ndim,
            nnodes,
            nbottom,
            ninterior: nbottom - 1,
            nlevels: maxlevel,
        };
        assert_eq!(metadata.nbottom + metadata.ninterior, metadata.nnodes);
        KDTree {
            metadata,
            lr: todo!(),
            perm: todo!(),
            cut: todo!(),
            data: todo!(),
            splitdim: todo!(),
            minval: todo!(),
            maxval: todo!(),
            scale: todo!(),
            invscale: todo!(),
        }
    }

    pub fn build(
        metadata: KdtMetadata,
        data: &[D],
        build_options: BuildOptions,
    ) -> KDTree<'a, T, D> {
        todo!()
    }

    pub fn set_limits(&self, low: f64, high: f64) {
        todo!()
    }

    pub fn permute(&mut self, ind: usize) {
        todo!()
    }

    pub fn inverse_permute(&self) -> Option<usize> {
        todo!()
    }

    pub fn check() -> Result<(), Error> {
        todo!()
    }

    pub fn copy_data_double() {
        todo!("just provide a ref to data?")
    }

    pub fn get_data() {
        todo!("just provide a ref to data?")
    }
    pub fn kdtype_parse_data_string() {
        todo!("just use KdtMetadata struct?")
    }
    pub fn kdtype_parse_tree_string() {
        todo!("just use KdtMetadata struct?")
    }
    pub fn kdtypes_to_treetype() {
        todo!("just use KdtMetadata struct?")
    }

    /* Nearest neighbour: returns the index _in the kdtree_ of the nearest point;
     * the point is at  (kd->data + ind * kd->ndim)  and its permuted index is
     * (kd->perm[ind]).
     *
     * Returns the distance-squared to the nearest neighbour
     * will be placed there. (optionally?)
     */
    pub fn nearest_neighbour(&self, point: usize) -> (usize, f64) {
        todo!()
    }
    /* Nearest neighbour (if within a maximum range): returns the index
     * _in the kdtree_ of the nearest point, _if_ its distance is less than
     * maxd2.  (Otherwise, -1).
     *
     * Returns the distance-squared to the nearest neighbour
     * will be placed there. (optionally?)
     */
    pub fn nearest_neighbour_within(&self, point: usize, max_dist: f64) -> (usize, f64) {
        todo!()
    }

    pub fn rangesearch(&self, point: usize, max_dist: f64) -> () {
        todo!()
    }

    fn is_leaf(&self, nodeid: usize) -> bool {
        nodeid >= self.metadata.ninterior
    }

    fn get_left_node(&self, nodeid: usize) -> usize {
        if self.is_leaf(nodeid) {
            self.leaf_left(nodeid)
        } else {
            let leftmost = self.first_leaf(nodeid);
            self.leaf_left(leftmost)
        }
    }

    fn get_right_node(&self, nodeid: usize) -> usize {
        if self.is_leaf(nodeid) {
            self.leaf_right(nodeid)
        } else {
            let rightmost = self.last_leaf(nodeid);
            self.leaf_right(rightmost)
        }
    }

    fn leaf_left(&self, nodeid: usize) -> usize {
        let leafid = nodeid - self.metadata.ninterior;
        if leafid == 0 {
            0
        } else if self.metadata.has_linear_lr {
            self.linear_lr(leafid)
        } else if !self.lr.is_empty() {
            (self.lr[leafid - 1] + 1) as usize
        } else {
            self.calculate_R(leafid - 1) + 1
        }
    }

    fn leaf_right(&self, nodeid: usize) -> usize {
        let leafid = nodeid - self.metadata.ninterior;
        if self.metadata.has_linear_lr {
            self.linear_lr(leafid + 1) - 1
        } else if !self.lr.is_empty() {
            self.lr[leafid] as usize
        } else {
            self.calculate_R(leafid)
        }
    }
    fn first_leaf(&self, nodeid: usize) -> usize {
        let dlevel = self.metadata.nlevels - 1 - node_level(nodeid);
        ((nodeid + 1) << dlevel) - 1
    }
    fn last_leaf(&self, nodeid: usize) -> usize {
        let dlevel = self.metadata.nlevels - 1 - node_level(nodeid);
        let twodl = 1 << dlevel;
        let nodeid_twodl = nodeid << dlevel;
        nodeid_twodl + (twodl - 1) * 2
    }
    fn linear_lr(&self, nodeid: usize) -> usize {
        (nodeid * self.metadata.ndata) / self.metadata.nbottom
    }
    fn calculate_R(&self, nodeid: usize) -> usize {
        let mut mask = 1 << (self.metadata.nlevels - 1);
        let mut L = 0;
        let mut N = self.metadata.ndata;
        let nextguy = nodeid + 1;
        if nextguy == self.metadata.nbottom {
            return self.metadata.ndata - 1;
        }
        for l in 0..(self.metadata.nlevels - 1) {
            mask /= 2;
            if (nextguy & mask) > 0 {
                L += N / 2;
                N = (N + 1) / 2;
            } else {
                N = N / 2;
            }
        }
        L - 1
    }
}

pub(crate) enum TreeCut<'a, T> {
    BoundingBox(&'a [T]),
    SplitDim { data: &'a [T], mask: DimSplit },
}

#[derive(Debug)]
pub(crate) struct KDRange {
    pub(crate) range: Vec<(f64, f64)>,
    pub(crate) scale: f64,
    pub(crate) invscale: f64,
}

pub(crate) struct DimSplit {
    pub(crate) dimbits: u8,
    pub(crate) dimmask: u32,
    pub(crate) splitmask: u32,
}

pub struct BuildOptions {
    bbox: bool,
    split: bool,
    /* Only applicable to integer trees: use a separate array to hold the
    splitting dimension, rather than packing it into the bottom bits
    of the splitting plane location. */
    split_dim: bool,
    no_lr: bool,
    /* Twiddle the split locations so that computing LR is O(1).
    Only works for double trees or int trees with KD_BUILD_SPLITDIM. */
    linear_lr: bool,
}
