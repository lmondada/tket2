#![allow(non_snake_case)]

//! Tuple implementations for the [`Rewriter`] trait.
//!
//! This module provides implementations of [`Rewriter`] for tuples of rewriters,
//! enabling elegant composition of multiple rewriters.

use super::{CircuitRewrite, Rewriter};
use crate::Circuit;
use hugr::HugrView;

/// Macro to generate Rewriter implementations for tuples of various sizes.
macro_rules! impl_rewriter_for_tuples {
    // Generate implementation for a specific tuple size
    ($(($($generic:ident),+)),+) => {
        $(
            impl<N, $($generic),+> Rewriter<N> for ($($generic,)+)
            where
                $($generic: Rewriter<N>,)+
            {
                fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
                    let mut rewrites = Vec::new();
                    let ($($generic,)+) = self;
                    $(
                        rewrites.extend($generic.get_rewrites(circ));
                    )+
                    rewrites
                }
            }
        )+
    };
}

// Generate implementations for tuples of size 2 through 8
impl_rewriter_for_tuples! {
    (R1, R2),
    (R1, R2, R3),
    (R1, R2, R3, R4),
    (R1, R2, R3, R4, R5),
    (R1, R2, R3, R4, R5, R6),
    (R1, R2, R3, R4, R5, R6, R7),
    (R1, R2, R3, R4, R5, R6, R7, R8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TketOp;
    use hugr::{
        builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        hugr::views::SiblingSubgraph,
        ops::handle::DfgID,
    };

    /// Mock rewriter for testing that returns a specific number of empty rewrites
    #[derive(Clone, Debug)]
    struct MockRewriter;

    impl Rewriter<hugr::Node> for MockRewriter {
        fn get_rewrites(
            &self,
            circ: &Circuit<impl HugrView<Node = hugr::Node>>,
        ) -> Vec<CircuitRewrite<hugr::Node>> {
            // Return a single (dummy) rewrite
            vec![CircuitRewrite::try_new(
                &SiblingSubgraph::try_new_dataflow_subgraph::<_, DfgID>(circ.to_owned().hugr())
                    .unwrap(),
                circ.hugr(),
                circ.to_owned(),
            )
            .unwrap()]
        }
    }

    fn create_test_circuit() -> Circuit {
        let mut h = DFGBuilder::new(endo_sig(qb_t())).unwrap();
        let qbs = h.input_wires();
        let mut circ = h.as_circuit(qbs);
        circ.append(TketOp::H, [0]).unwrap();
        let qbs = circ.finish();
        Circuit::new(h.finish_hugr_with_outputs(qbs).unwrap())
    }

    use rstest::rstest;

    #[rstest]
    #[case((MockRewriter, MockRewriter), 2)]
    #[case((MockRewriter, MockRewriter, MockRewriter), 3)]
    #[case((MockRewriter, MockRewriter, MockRewriter, MockRewriter), 4)]
    #[case((MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter), 5)]
    #[case((MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter), 6)]
    #[case((MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter), 7)]
    #[case((MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter, MockRewriter), 8)]
    fn test_tuple_rewriters_empty_vecs<R>(#[case] rewriter: R, #[case] expected_len: usize)
    where
        R: Rewriter<hugr::Node>,
    {
        let circuit = create_test_circuit();
        let rewrites = rewriter.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), expected_len);
    }
}
