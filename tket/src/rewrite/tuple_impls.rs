//! Tuple implementations for the [`Rewriter`] trait.
//!
//! This module provides implementations of [`Rewriter`] for tuples of rewriters,
//! enabling elegant composition of multiple rewriters.

use super::{CircuitRewrite, Rewriter};
use crate::Circuit;
use hugr::{HugrView, Node};

/// Macro to generate Rewriter implementations for tuples of various sizes.
///
/// This generates implementations for tuples containing 2-8 rewriters.
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
    use hugr::{
        builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
    };
    use crate::TketOp;

    /// Mock rewriter for testing that returns a specific number of empty rewrites
    #[derive(Clone, Debug)]
    struct MockRewriter(usize);

    impl Rewriter<Node> for MockRewriter {
        fn get_rewrites(&self, _circ: &Circuit<impl HugrView<Node = Node>>) -> Vec<CircuitRewrite<Node>> {
            // Return a number of empty rewrites equal to our ID for testing
            vec![/* empty for testing */; self.0]
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

    #[test]
    fn test_tuple_2() {
        let circuit = create_test_circuit();
        let rewriter = (MockRewriter(1), MockRewriter(2));
        let rewrites = rewriter.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), 3); // 1 + 2 = 3 empty rewrites
    }

    #[test]
    fn test_tuple_3() {
        let circuit = create_test_circuit();
        let rewriter = (MockRewriter(1), MockRewriter(2), MockRewriter(3));
        let rewrites = rewriter.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), 6); // 1 + 2 + 3 = 6
    }

    #[test]
    fn test_tuple_4() {
        let circuit = create_test_circuit();
        let rewriter = (MockRewriter(1), MockRewriter(1), MockRewriter(1), MockRewriter(1));
        let rewrites = rewriter.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), 4); // 1 + 1 + 1 + 1 = 4
    }

    #[test]
    fn test_tuple_5() {
        let circuit = create_test_circuit();
        let rewriter = (
            MockRewriter(1), MockRewriter(1), MockRewriter(1), 
            MockRewriter(1), MockRewriter(1)
        );
        let rewrites = rewriter.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), 5); // 5 × 1 = 5
    }

    #[test]
    fn test_tuple_6() {
        let circuit = create_test_circuit();
        let rewriter = (
            MockRewriter(1), MockRewriter(1), MockRewriter(1), 
            MockRewriter(1), MockRewriter(1), MockRewriter(1)
        );
        let rewrites = rewriter.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), 6); // 6 × 1 = 6
    }

    #[test]
    fn test_tuple_7() {
        let circuit = create_test_circuit();
        let rewriter = (
            MockRewriter(1), MockRewriter(1), MockRewriter(1), MockRewriter(1),
            MockRewriter(1), MockRewriter(1), MockRewriter(1)
        );
        let rewrites = rewriter.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), 7); // 7 × 1 = 7
    }

    #[test]
    fn test_tuple_8() {
        let circuit = create_test_circuit();
        let rewriter = (
            MockRewriter(1), MockRewriter(1), MockRewriter(1), MockRewriter(1),
            MockRewriter(1), MockRewriter(1), MockRewriter(1), MockRewriter(1),
        );
        let rewrites = rewriter.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), 8); // 8 × 1 = 8
    }

    #[test]
    fn test_mixed_tuple_sizes() {
        let circuit = create_test_circuit();
        
        // Test that different rewriter combinations sum correctly
        let rewriter_mixed = (MockRewriter(2), MockRewriter(3), MockRewriter(5));
        let rewrites = rewriter_mixed.get_rewrites(&circuit);
        assert_eq!(rewrites.len(), 10); // 2 + 3 + 5 = 10
    }
}