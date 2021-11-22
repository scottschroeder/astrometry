use super::KDTree;
use crate::kdtree::error::Error;
use crate::kdtree::kd_types::IsInteger;
use crate::kdtree::tree::TreeCut;
use num_traits::Bounded;
use num_traits::Num;
use num_traits::NumCast;
use num_traits::NumOps;
use num_traits::ToPrimitive;
use std::borrow::Cow;

const NODESTACK_DEFAULT_SIZE: usize = 100;

pub struct Query<'a, D>
where
    [D]: ToOwned<Owned = Vec<D>>,
{
    inner: Cow<'a, [D]>,
}

impl<'a, D> Query<'a, D>
where
    D: Clone,
    [D]: ToOwned<Owned = Vec<D>>,
{
    pub fn new<T>(inner: T) -> Query<'a, D>
    where
        Cow<'a, [D]>: From<T>,
    {
        Query {
            inner: Cow::from(inner),
        }
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a, D> AsRef<[D]> for Query<'a, D>
where
    [D]: ToOwned<Owned = Vec<D>>,
{
    fn as_ref(&self) -> &[D] {
        self.inner.as_ref()
    }
}

pub struct RangeSearchOptions {
    compute_dists: bool,
    return_points: bool,
    sort_dists: bool,
    small_radius: bool,
    /* If both bounding box and splitting plane are available,
    use splitting plane (default is bounding box)... */
    use_split: bool,
    /* If the tree is u32 and bounding boxes are being used and the
    rangesearch only fits in 64 bits, use doubles instead of u64 in
    the distance computation (default is to use u64).
    (Likewise for u16/u32)
    */
    no_big_int_math: bool,
    /*
    In bounding-box trees that also have a "splitdim" array,
    do a quick check along the splitting dimension.
    */
    split_precheck: bool,
    /*
    In integer bounding-box trees, do an L1 distance pre-check.
    */
    l1_precheck: bool,
    /*
    Don't resize the kdtree_qres_t* result structure to take only the
    space required (assume it's going to be reused and we're letting the
    memory usage do the "high water mark" thing).
    */
    no_resize_results: bool,
}

impl RangeSearchOptions {
    fn do_dists(&self) -> bool {
        self.compute_dists || self.sort_dists
    }
}

#[derive(Debug)]
pub struct QueryResult<'a, D> {
    distance: f64,
    data: &'a [D],
    index: usize,
}

#[derive(Debug)]
pub struct QueryResults<'a, D> {
    inner: Vec<QueryResult<'a, D>>,
}

impl<'a, D> Default for QueryResults<'a, D> {
    fn default() -> Self {
        Self { inner: Vec::new() }
    }
}

impl<'a, D> QueryResults<'a, D> {
    fn add_result(&mut self, index: usize, sdist: f64, pt: &'a [D]) {
        self.inner.push(QueryResult {
            distance: sdist,
            data: pt,
            index,
        })
    }
}

pub trait SearchableTreeType: IsInteger + NumCast + Num + Bounded {}
pub trait DataType: ToPrimitive + Clone {}
pub trait QueryType: ToPrimitive + NumCast + NumOps + PartialOrd + Clone {}

impl<'a, T, D> KDTree<'a, T, D>
where
    T: SearchableTreeType,
    D: DataType,
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub fn rangesearch_options<E>(
        &self,
        query: Query<'_, E>,
        maxd2: f64,
        options: RangeSearchOptions,
    ) -> Result<(), Error>
    where
        E: QueryType,
        [E]: ToOwned<Owned = Vec<E>>,
    {
        let mut nodestack: Vec<usize> = Vec::with_capacity(NODESTACK_DEFAULT_SIZE);

        // TODO rename D to ndim?
        let D = self.metadata.ndim;

        let do_dists = options.do_dists();
        if !do_dists {
            log::warn!("going to ignore do_dists=false, and do them anyway");
        }
        let do_points = true;
        let mut maxdist = 0.0;
        let mut tlinf = T::zero();
        let mut tl1 = T::zero();
        let mut tl2 = T::zero();
        let mut bigtl2 = T::zero();

        let mut use_tquery = false;
        let mut use_tsplit = false;
        let mut use_tmath = false;
        // TODO bigtmath unsupported
        let use_bigtmath = false;

        let use_bboxes = self.bb_any();
        let use_splits = !self.bb_any();

        let mut dtl1 = 0.0;
        let mut dtl2 = 0.0;
        let mut dtlinf = 0.0;

        let mut tquery = None;

        let do_wholenode_check = !options.small_radius;

        let mut do_precheck = options.split_precheck && self.bb_any();
        let mut do_l1precheck = options.l1_precheck && self.bb_any();

        maxdist = maxd2.sqrt();

        let ttype_max = <f64 as NumCast>::from(T::max_value()).unwrap();
        let ttype_min = <f64 as NumCast>::from(T::min_value()).unwrap();

        if T::is_integer() && (!self.bb_any() || do_precheck || do_l1precheck) {
            tquery = self.ttype_query(&query)
        }
        if tquery.is_some() {
            dtl1 = maxdist * (D as f64).sqrt();
            dtl2 = maxd2;
            dtlinf = maxdist;
            tl1 = T::from(dtl1.ceil()).unwrap();
            tlinf = T::from(dtlinf.ceil()).unwrap();
            bigtl2 = T::from(dtl2.ceil()).unwrap();
            use_tsplit = dtlinf < <f64 as NumCast>::from(T::max_value()).unwrap();
        }

        if do_l1precheck {
            if dtl1 > ttype_max {
                do_l1precheck = false
            }
        }
        if T::is_integer() && tquery.is_some() && self.bb_any() {
            if dtl2 < ttype_max {
                use_tmath = true;
                // else if use bigttype else warn overflow?
            }
        }

        nodestack.push(0);

        let mut results = QueryResults::default();

        while let Some(nodeid) = nodestack.pop() {
            let mut split = T::zero();
            let mut dim = None;
            // let tlo = None;
            // let thi = None;
            if self.is_leaf(nodeid) {
                let L = self.leaf_left(nodeid);
                let R = self.leaf_right(nodeid);

                for i in L..R {
                    let data = &self.data[D * i..D * (i + 1)];
                    if let Some(dqsd) = self.dist2_bailout(&query, data, maxd2) {
                        results.add_result(i, dqsd, data);
                    } else {
                        continue;
                    }
                }
                continue;
            }

            if !self.splitdim.is_empty() {
                dim = Some(self.splitdim[nodeid])
            }

            match &self.cut {
                crate::kdtree::tree::TreeCut::BoundingBox(b) => {
                    assert!(
                        use_bboxes,
                        "only have bounding boxes, but told not to use them"
                    );
                    todo!("I don't know how to use bboxes")
                }
                crate::kdtree::tree::TreeCut::SplitDim { data, mask } => {
                    assert!(
                        !use_bboxes,
                        "told to use bounding boxes, but I don't have them"
                    );
                    let mut split = &data[nodeid];
                    if self.splitdim.is_empty() && T::is_integer() {
                        let tmpsplit = split;
                        dim = Some(tmpsplit & mask.dimmask);
                        split = tmpsplit & mask.splitmask;
                        // Clipboard

                    }
                }
            }
        }

        todo!()
    }
    fn ttype_query<'b, E>(&self, query: &Query<'b, E>) -> Option<Query<'b, T>>
    where
        E: ToPrimitive + Clone,
        [E]: ToOwned<Owned = Vec<E>>,
        [T]: ToOwned<Owned = Vec<T>>,
    {
        let mut tquery: Vec<T> = Vec::with_capacity(query.len());
        for q in query.as_ref() {
            if let Some(t) = check_convert(q.clone()) {
                tquery.push(t)
            } else {
                return None;
            }
        }
        return Some(Query {
            inner: Cow::Owned(tquery),
        });
    }
    fn dist2_bailout<'b, E>(&self, q: &Query<'b, E>, p: &[D], maxd2: f64) -> Option<f64>
    where
        E: NumCast + Clone + NumOps,
    {
        let mut d2 = 0.0f64;
        for (dd, qq) in p.iter().zip(q.as_ref()) {
            let pp: E = check_convert(dd.clone()).unwrap();
            let delta = qq.clone() - pp;
            let delta2: f64 = check_convert(delta.clone() * delta).unwrap();
            d2 += delta2;
            if d2 > maxd2 {
                return None;
            }
        }
        Some(d2)
    }
    fn dist2_exceeds<'b, E>(&self, q: &Query<'b, E>, p: &[D], maxd2: f64) -> bool
    where
        E: NumCast + Clone + NumOps + PartialOrd,
    {
        let mut d2 = 0.0f64;
        for (dd, qq) in p.iter().zip(q.as_ref()) {
            let pp: E = check_convert(dd.clone()).unwrap();
            let qq = qq.clone();
            let delta = if T::is_integer() {
                if qq > pp {
                    qq - pp
                } else {
                    pp - qq
                }
            } else {
                qq - pp
            };
            let delta2: f64 = check_convert(delta.clone() * delta).unwrap();
            d2 += delta2;
            if d2 > maxd2 {
                return true;
            }
        }
        false
    }
}

fn check_convert<S, D>(src: S) -> Option<D>
where
    D: NumCast,
    S: ToPrimitive,
{
    D::from(src)
}