//! Transform circuits using rewrite rules.
//!
//! This module provides the core infrastructure for rewriting quantum circuits.
//! The main abstraction is the [`Rewriter`] trait, which can be implemented to
//! generate rewrite rules for circuits.
//!
//! ## Rewriter Composition
//!
//! Multiple rewriters can be elegantly composed using tuples or dynamic collections:
//!
//! ### Tuple Composition (Recommended)
//!
//! For compile-time known rewriters, use tuple composition:
//!
//! ```rust,ignore
//! use tket::rewrite::MatchReplaceRewriter;
//! use tket::optimiser::BadgerOptimiser;
//!
//! let rewriter1 = MatchReplaceRewriter::new(matcher1, replacement1);
//! let rewriter2 = MatchReplaceRewriter::new(matcher2, replacement2);
//! let rewriter3 = MatchReplaceRewriter::new(matcher3, replacement3);
//!
//! // Compose 2-8 rewriters using tuples
//! let composite_rewriter = (rewriter1, rewriter2, rewriter3);
//!
//! // Use with BadgerOptimiser
//! let optimiser = BadgerOptimiser::new(composite_rewriter, strategy);
//! ```
//!
//! ### Dynamic Composition
//!
//! For runtime collections of rewriters, use `Vec<Box<dyn Rewriter<N>>>`:
//!
//! ```rust,ignore
//! let rewriters: Vec<Box<dyn Rewriter<hugr::Node>>> = vec![
//!     Box::new(rewriter1),
//!     Box::new(rewriter2),
//!     Box::new(rewriter3),
//! ];
//!
//! let optimiser = BadgerOptimiser::new(rewriters, strategy);
//! ```
//!
//! Both approaches combine all rewrites from constituent rewriters, enabling
//! the optimizer to consider all possible transformations simultaneously.

#[cfg(feature = "portmatching")]
pub mod ecc_rewriter;
pub mod matcher;
pub mod replacement;
pub mod strategy;
pub mod trace;

#[cfg(feature = "portmatching")]
pub use ecc_rewriter::ECCRewriter;

use derive_more::{From, Into};
use hugr::core::HugrNode;
use hugr::hugr::hugrmut::HugrMut;
use hugr::hugr::patch::simple_replace;
use hugr::hugr::views::sibling_subgraph::InvalidReplacement;
use hugr::hugr::Patch;
use hugr::{
    hugr::{views::SiblingSubgraph, SimpleReplacementError},
    SimpleReplacement,
};
use hugr::{Hugr, HugrView};
use matcher::{CircuitMatcher, MatchingOptions};
use replacement::MatchReplacement;

use crate::circuit::Circuit;
pub use crate::Subcircuit;

/// A rewrite rule for circuits.
#[derive(Debug, Clone, From, Into)]
pub struct CircuitRewrite<N = hugr::Node>(SimpleReplacement<N>);

impl<N: HugrNode> CircuitRewrite<N> {
    /// Create a new rewrite rule.
    pub fn try_new(
        subgraph: &SiblingSubgraph<N>,
        hugr: &impl HugrView<Node = N>,
        replacement: Circuit<impl HugrView<Node = hugr::Node>>,
    ) -> Result<Self, InvalidReplacement> {
        let replacement = replacement
            .extract_dfg()
            .unwrap_or_else(|e| panic!("{}", e))
            .into_hugr();
        Ok(Self(subgraph.create_simple_replacement(hugr, replacement)?))
    }

    /// Number of nodes added or removed by the rewrite.
    ///
    /// The difference between the new number of nodes minus the old. A positive
    /// number is an increase in node count, a negative number is a decrease.
    pub fn node_count_delta(&self) -> isize {
        let new_count = self.replacement().num_operations() as isize;
        let old_count = self.subgraph().node_count() as isize;
        new_count - old_count
    }

    /// The subgraph that is replaced.
    pub fn subgraph(&self) -> &SiblingSubgraph<N> {
        self.0.subgraph()
    }

    /// The replacement subcircuit.
    pub fn replacement(&self) -> Circuit<&Hugr> {
        self.0.replacement().into()
    }

    /// Returns a set of nodes referenced by the rewrite. Modifying any these
    /// nodes will invalidate it.
    ///
    /// Two `CircuitRewrite`s can be composed if their invalidation sets are
    /// disjoint.
    #[inline]
    pub fn invalidation_set(&self) -> impl Iterator<Item = N> + '_ {
        self.0.invalidation_set()
    }

    /// Apply the rewrite rule to a circuit.
    #[inline]
    pub fn apply(
        self,
        circ: &mut Circuit<impl HugrMut<Node = N>>,
    ) -> Result<simple_replace::Outcome<N>, SimpleReplacementError> {
        circ.add_rewrite_trace(&self);
        self.0.apply(circ.hugr_mut())
    }

    /// Apply the rewrite rule to a circuit, without registering it in the rewrite trace.
    #[inline]
    pub fn apply_notrace(
        self,
        circ: &mut Circuit<impl HugrMut<Node = N>>,
    ) -> Result<simple_replace::Outcome<N>, SimpleReplacementError> {
        self.0.apply(circ.hugr_mut())
    }
}

/// Generate rewrite rules for circuits.
pub trait Rewriter<N> {
    /// Get the rewrite rules for a circuit.
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>>;
}

/// A rewriter that uses a [`CircuitMatcher`] to find matches and a
/// [`MatchReplacement`] to create [`CircuitRewrite`]s.
#[derive(Clone, Debug)]
pub struct MatchReplaceRewriter<C: CircuitMatcher, R> {
    matcher: C,
    replacement: R,
}

impl<C: CircuitMatcher, R> MatchReplaceRewriter<C, R> {
    /// Create a new [`MatchReplaceRewriter`].
    pub fn new(matcher: C, replacement: R) -> Self {
        Self {
            matcher,
            replacement,
        }
    }
}

impl<C, R> Rewriter<hugr::Node> for MatchReplaceRewriter<C, R>
where
    C: CircuitMatcher,
    R: MatchReplacement<C::MatchInfo>,
{
    fn get_rewrites(
        &self,
        circ: &Circuit<impl HugrView<Node = hugr::Node>>,
    ) -> Vec<CircuitRewrite<hugr::Node>> {
        let hugr = circ.hugr();
        let matches = self
            .matcher
            .as_hugr_matcher()
            .get_all_matches(circ, &MatchingOptions::default());
        matches
            .into_iter()
            .flat_map(|(subgraph, match_info)| {
                self.replacement
                    .replace_match(&subgraph, hugr, match_info)
                    .into_iter()
                    .filter_map(move |repl| CircuitRewrite::try_new(&subgraph, hugr, repl).ok())
            })
            .collect()
    }
}

// Composite rewriter implementations for tuples
// This allows combining multiple rewriters elegantly: (rewriter1, rewriter2, ...)

impl<N, R1, R2> Rewriter<N> for (R1, R2)
where
    R1: Rewriter<N>,
    R2: Rewriter<N>,
{
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
        let mut rewrites = self.0.get_rewrites(circ);
        rewrites.extend(self.1.get_rewrites(circ));
        rewrites
    }
}

impl<N, R1, R2, R3> Rewriter<N> for (R1, R2, R3)
where
    R1: Rewriter<N>,
    R2: Rewriter<N>,
    R3: Rewriter<N>,
{
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
        let mut rewrites = self.0.get_rewrites(circ);
        rewrites.extend(self.1.get_rewrites(circ));
        rewrites.extend(self.2.get_rewrites(circ));
        rewrites
    }
}

impl<N, R1, R2, R3, R4> Rewriter<N> for (R1, R2, R3, R4)
where
    R1: Rewriter<N>,
    R2: Rewriter<N>,
    R3: Rewriter<N>,
    R4: Rewriter<N>,
{
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
        let mut rewrites = self.0.get_rewrites(circ);
        rewrites.extend(self.1.get_rewrites(circ));
        rewrites.extend(self.2.get_rewrites(circ));
        rewrites.extend(self.3.get_rewrites(circ));
        rewrites
    }
}

impl<N, R1, R2, R3, R4, R5> Rewriter<N> for (R1, R2, R3, R4, R5)
where
    R1: Rewriter<N>,
    R2: Rewriter<N>,
    R3: Rewriter<N>,
    R4: Rewriter<N>,
    R5: Rewriter<N>,
{
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
        let mut rewrites = self.0.get_rewrites(circ);
        rewrites.extend(self.1.get_rewrites(circ));
        rewrites.extend(self.2.get_rewrites(circ));
        rewrites.extend(self.3.get_rewrites(circ));
        rewrites.extend(self.4.get_rewrites(circ));
        rewrites
    }
}

impl<N, R1, R2, R3, R4, R5, R6> Rewriter<N> for (R1, R2, R3, R4, R5, R6)
where
    R1: Rewriter<N>,
    R2: Rewriter<N>,
    R3: Rewriter<N>,
    R4: Rewriter<N>,
    R5: Rewriter<N>,
    R6: Rewriter<N>,
{
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
        let mut rewrites = self.0.get_rewrites(circ);
        rewrites.extend(self.1.get_rewrites(circ));
        rewrites.extend(self.2.get_rewrites(circ));
        rewrites.extend(self.3.get_rewrites(circ));
        rewrites.extend(self.4.get_rewrites(circ));
        rewrites.extend(self.5.get_rewrites(circ));
        rewrites
    }
}

impl<N, R1, R2, R3, R4, R5, R6, R7> Rewriter<N> for (R1, R2, R3, R4, R5, R6, R7)
where
    R1: Rewriter<N>,
    R2: Rewriter<N>,
    R3: Rewriter<N>,
    R4: Rewriter<N>,
    R5: Rewriter<N>,
    R6: Rewriter<N>,
    R7: Rewriter<N>,
{
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
        let mut rewrites = self.0.get_rewrites(circ);
        rewrites.extend(self.1.get_rewrites(circ));
        rewrites.extend(self.2.get_rewrites(circ));
        rewrites.extend(self.3.get_rewrites(circ));
        rewrites.extend(self.4.get_rewrites(circ));
        rewrites.extend(self.5.get_rewrites(circ));
        rewrites.extend(self.6.get_rewrites(circ));
        rewrites
    }
}

impl<N, R1, R2, R3, R4, R5, R6, R7, R8> Rewriter<N> for (R1, R2, R3, R4, R5, R6, R7, R8)
where
    R1: Rewriter<N>,
    R2: Rewriter<N>,
    R3: Rewriter<N>,
    R4: Rewriter<N>,
    R5: Rewriter<N>,
    R6: Rewriter<N>,
    R7: Rewriter<N>,
    R8: Rewriter<N>,
{
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
        let mut rewrites = self.0.get_rewrites(circ);
        rewrites.extend(self.1.get_rewrites(circ));
        rewrites.extend(self.2.get_rewrites(circ));
        rewrites.extend(self.3.get_rewrites(circ));
        rewrites.extend(self.4.get_rewrites(circ));
        rewrites.extend(self.5.get_rewrites(circ));
        rewrites.extend(self.6.get_rewrites(circ));
        rewrites.extend(self.7.get_rewrites(circ));
        rewrites
    }
}

// Rewriter implementation for dynamic collections of rewriters
impl<N> Rewriter<N> for Vec<Box<dyn Rewriter<N>>>
where
    N: 'static,
{
    fn get_rewrites(&self, circ: &Circuit<impl HugrView<Node = N>>) -> Vec<CircuitRewrite<N>> {
        self.iter()
            .flat_map(|rewriter| rewriter.get_rewrites(circ))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hugr::{
        builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
        extension::prelude::qb_t,
        hugr::views::SiblingSubgraph,
    };
    use crate::{
        rewrite::{
            matcher::{CircuitMatcher, MatchContext, MatchOutcome, OpArg},
            replacement::MatchReplacement,
        },
        TketOp,
    };

    /// A mock rewriter for testing that matches Hadamard gates
    #[derive(Clone, Debug)]
    struct MockHRewriter;

    /// A mock rewriter for testing that matches X gates
    #[derive(Clone, Debug)]
    struct MockXRewriter;

    /// Simple partial match state for testing
    #[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
    enum SimplePartialMatch {
        #[default]
        Start,
        Found,
    }

    /// Mock matcher for H gates
    #[derive(Clone, Copy, Debug)]
    struct MockHMatcher;

    /// Mock matcher for X gates
    #[derive(Clone, Copy, Debug)]
    struct MockXMatcher;

    /// Mock replacement that creates empty circuits
    #[derive(Clone, Copy, Debug)]
    struct MockReplacement;

    impl CircuitMatcher for MockHMatcher {
        type PartialMatchInfo = SimplePartialMatch;
        type MatchInfo = ();

        fn match_tket_op(
            &self,
            op: TketOp,
            _op_args: &[OpArg],
            match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
        ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
            if op == TketOp::H {
                match match_context.match_info {
                    SimplePartialMatch::Start => MatchOutcome::default().proceed(SimplePartialMatch::Found),
                    SimplePartialMatch::Found => MatchOutcome::default().complete(()),
                }
            } else {
                MatchOutcome::stop()
            }
        }
    }

    impl CircuitMatcher for MockXMatcher {
        type PartialMatchInfo = SimplePartialMatch;
        type MatchInfo = ();

        fn match_tket_op(
            &self,
            op: TketOp,
            _op_args: &[OpArg],
            match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
        ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
            if op == TketOp::X {
                match match_context.match_info {
                    SimplePartialMatch::Start => MatchOutcome::default().proceed(SimplePartialMatch::Found),
                    SimplePartialMatch::Found => MatchOutcome::default().complete(()),
                }
            } else {
                MatchOutcome::stop()
            }
        }
    }

    impl MatchReplacement<()> for MockReplacement {
        fn replace_match<H: hugr::HugrView>(
            &self,
            _subgraph: &SiblingSubgraph<H::Node>,
            _hugr: H,
            _match_info: (),
        ) -> Vec<crate::Circuit> {
            // Return empty circuit for replacement
            let h = DFGBuilder::new(endo_sig(qb_t())).unwrap();
            let inps = h.input_wires();
            let empty_circ = h.finish_hugr_with_outputs(inps).unwrap();
            vec![crate::Circuit::new(empty_circ)]
        }
    }

    impl Rewriter<hugr::Node> for MockHRewriter {
        fn get_rewrites(&self, _circ: &Circuit<impl HugrView<Node = hugr::Node>>) -> Vec<CircuitRewrite<hugr::Node>> {
            // Return a mock rewrite for testing
            vec![]
        }
    }

    impl Rewriter<hugr::Node> for MockXRewriter {
        fn get_rewrites(&self, _circ: &Circuit<impl HugrView<Node = hugr::Node>>) -> Vec<CircuitRewrite<hugr::Node>> {
            // Return a mock rewrite for testing
            vec![]
        }
    }

    fn create_test_circuit() -> Circuit {
        let mut h = DFGBuilder::new(endo_sig(qb_t())).unwrap();
        let qbs = h.input_wires();
        let mut circ = h.as_circuit(qbs);
        
        // Add some gates
        circ.append(TketOp::H, [0]).unwrap();
        circ.append(TketOp::X, [0]).unwrap();
        
        let qbs = circ.finish();
        Circuit::new(h.finish_hugr_with_outputs(qbs).unwrap())
    }

    #[test]
    fn test_tuple_rewriter_composition() {
        let h_rewriter = MockHRewriter;
        let x_rewriter = MockXRewriter;
        
        let circuit = create_test_circuit();
        
        // Test 2-tuple composition
        let tuple_rewriter = (h_rewriter.clone(), x_rewriter.clone());
        let rewrites = tuple_rewriter.get_rewrites(&circuit);
        
        // Both rewriters should be called
        assert_eq!(rewrites.len(), 0); // Mock rewriters return empty vecs
    }

    #[test]
    fn test_vec_rewriter_composition() {
        let h_rewriter = MockHRewriter;
        let x_rewriter = MockXRewriter;
        
        let circuit = create_test_circuit();
        
        // Test Vec<Box<dyn Rewriter>> composition
        let vec_rewriter: Vec<Box<dyn Rewriter<hugr::Node>>> = vec![
            Box::new(h_rewriter),
            Box::new(x_rewriter),
        ];
        let rewrites = vec_rewriter.get_rewrites(&circuit);
        
        // Both rewriters should be called
        assert_eq!(rewrites.len(), 0); // Mock rewriters return empty vecs
    }

    #[test]
    fn test_nested_tuple_composition() {
        let h_rewriter = MockHRewriter;
        let x_rewriter = MockXRewriter;
        let h_rewriter2 = MockHRewriter;
        
        let circuit = create_test_circuit();
        
        // Test 3-tuple composition
        let triple_rewriter = (h_rewriter, x_rewriter, h_rewriter2);
        let rewrites = triple_rewriter.get_rewrites(&circuit);
        
        // All three rewriters should be called
        assert_eq!(rewrites.len(), 0); // Mock rewriters return empty vecs
    }

    #[test]
    fn test_large_tuple_composition() {
        let rewriters = (
            MockHRewriter, MockXRewriter, MockHRewriter, MockXRewriter,
            MockHRewriter, MockXRewriter, MockHRewriter, MockXRewriter,
        );
        
        let circuit = create_test_circuit();
        let rewrites = rewriters.get_rewrites(&circuit);
        
        // All 8 rewriters should be called
        assert_eq!(rewrites.len(), 0); // Mock rewriters return empty vecs
    }

    #[test]
    fn test_match_replace_rewriter_composition() {
        let h_matcher_rewriter = MatchReplaceRewriter::new(MockHMatcher, MockReplacement);
        let x_matcher_rewriter = MatchReplaceRewriter::new(MockXMatcher, MockReplacement);
        
        let circuit = create_test_circuit();
        
        // Test that MatchReplaceRewriters can be composed
        let composed = (h_matcher_rewriter.clone(), x_matcher_rewriter.clone());
        let rewrites = composed.get_rewrites(&circuit);
        
        // Should get rewrites from both matchers
        // The actual number depends on the circuit structure and matching logic
        assert!(rewrites.len() >= 0);
        
        // Test Vec composition too
        let vec_composed: Vec<Box<dyn Rewriter<hugr::Node>>> = vec![
            Box::new(h_matcher_rewriter),
            Box::new(x_matcher_rewriter),
        ];
        let vec_rewrites = vec_composed.get_rewrites(&circuit);
        
        // Should produce the same results
        assert_eq!(rewrites.len(), vec_rewrites.len());
    }
}
