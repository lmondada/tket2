//! Using Badger to perform optimization with multiple rewriters.
//!
//! This example demonstrates how to combine multiple rewriters using tuple composition
//! for quantum circuit optimization with the Badger algorithm.

use hugr::{
    builder::{endo_sig, DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::qb_t,
    hugr::views::SiblingSubgraph,
};
use tket::{
    op_matches,
    optimiser::BadgerOptimiser,
    rewrite::{
        matcher::{CircuitMatcher, MatchContext, MatchOutcome, OpArg},
        replacement::MatchReplacement,
        strategy::LexicographicCostFunction,
        MatchReplaceRewriter,
    },
    Circuit, TketOp,
};

/// A matcher that matches two Hadamard gates in a row.
#[derive(Clone, Copy, Debug)]
struct TwoHMatcher;

/// A matcher that matches two X gates in a row.
#[derive(Clone, Copy, Debug)]
struct TwoXMatcher;

/// A replacement that replaces two Hadamard gates in a row with the identity.
#[derive(Clone, Copy, Debug)]
struct HadamardCancellation;

/// A replacement that replaces two X gates in a row with the identity.
#[derive(Clone, Copy, Debug)]
struct XCancellation;

/// State to keep track of how much has been matched so far.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
enum PartialMatchState {
    /// No gate matched so far.
    #[default]
    NoMatch,
    /// One gate matched so far.
    MatchedOne,
}

impl CircuitMatcher for TwoHMatcher {
    type PartialMatchInfo = PartialMatchState;
    type MatchInfo = ();

    fn match_tket_op(
        &self,
        op: tket::TketOp,
        _op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        if op != TketOp::H {
            // We are not interested in matching this op
            return MatchOutcome::stop();
        }

        match match_context.match_info {
            PartialMatchState::NoMatch => {
                // Making progress! Proceed to matching the second Hadamard
                MatchOutcome::default().proceed(PartialMatchState::MatchedOne)
            }
            PartialMatchState::MatchedOne => {
                // We have matched two Hadamards, so we can report the match
                MatchOutcome::default()
                    .complete(()) // report a full match
                    .skip(PartialMatchState::MatchedOne) // consider skipping this op (and match another one)
            }
        }
    }
}

impl CircuitMatcher for TwoXMatcher {
    type PartialMatchInfo = PartialMatchState;
    type MatchInfo = ();

    fn match_tket_op(
        &self,
        op: tket::TketOp,
        _op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        if op != TketOp::X {
            // We are not interested in matching this op
            return MatchOutcome::stop();
        }

        match match_context.match_info {
            PartialMatchState::NoMatch => {
                // Making progress! Proceed to matching the second X gate
                MatchOutcome::default().proceed(PartialMatchState::MatchedOne)
            }
            PartialMatchState::MatchedOne => {
                // We have matched two X gates, so we can report the match
                MatchOutcome::default()
                    .complete(()) // report a full match
                    .skip(PartialMatchState::MatchedOne) // consider skipping this op (and match another one)
            }
        }
    }
}

impl MatchReplacement<()> for HadamardCancellation {
    fn replace_match<H: hugr::HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        _match_info: (),
    ) -> Vec<tket::Circuit> {
        // subgraph should be a pair of Hadamards
        assert_eq!(subgraph.node_count(), 2);
        assert!(subgraph
            .nodes()
            .iter()
            .all(|&n| op_matches(hugr.get_optype(n), TketOp::H)));

        // The right hand side of the rewrite is just an empty one-qubit circuit
        let h = DFGBuilder::new(endo_sig(qb_t())).unwrap();
        let inps = h.input_wires();
        let empty_circ = h.finish_hugr_with_outputs(inps).unwrap();

        vec![Circuit::new(empty_circ)]
    }
}

impl MatchReplacement<()> for XCancellation {
    fn replace_match<H: hugr::HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        hugr: H,
        _match_info: (),
    ) -> Vec<tket::Circuit> {
        // subgraph should be a pair of X gates
        assert_eq!(subgraph.node_count(), 2);
        assert!(subgraph
            .nodes()
            .iter()
            .all(|&n| op_matches(hugr.get_optype(n), TketOp::X)));

        // The right hand side of the rewrite is just an empty one-qubit circuit
        let h = DFGBuilder::new(endo_sig(qb_t())).unwrap();
        let inps = h.input_wires();
        let empty_circ = h.finish_hugr_with_outputs(inps).unwrap();

        vec![Circuit::new(empty_circ)]
    }
}

/// A 4-qubit circuit made of three layers
///  - layer of 4x Hadamards on each qubit,
///  - layer of CX gates between pairs of qubits,
///  - layer of 4x Hadamards on each qubit,
///  - layer of 4x X gates on each qubit,
///  - layer of 4x X gates on each qubit (for cancellation),
fn mixed_gate_circuit() -> Circuit {
    let mut h = DFGBuilder::new(endo_sig(vec![qb_t(); 4])).unwrap();
    let qbs = h.input_wires();
    let mut circ = h.as_circuit(qbs);

    // Add Hadamards
    for _ in 0..4 {
        for i in 0..4 {
            circ.append(TketOp::H, [i]).unwrap();
        }
    }

    // Add CX gates
    for i in (0..4).step_by(2) {
        circ.append(TketOp::CX, [i, i + 1]).unwrap();
    }

    // Add more Hadamards
    for _ in 0..4 {
        for i in 0..4 {
            circ.append(TketOp::H, [i]).unwrap();
        }
    }

    // Add X gates that should cancel
    for _ in 0..2 {
        for i in 0..4 {
            circ.append(TketOp::X, [i]).unwrap();
        }
    }

    let qbs = circ.finish();
    Circuit::new(h.finish_hugr_with_outputs(qbs).unwrap())
}

fn main() {
    println!("=== Badger Optimization with Multiple Rewriters ===");
    
    // Create individual rewriters
    let hadamard_rewriter = MatchReplaceRewriter::new(TwoHMatcher, HadamardCancellation);
    let x_rewriter = MatchReplaceRewriter::new(TwoXMatcher, XCancellation);

    let circuit = mixed_gate_circuit();
    let initial_ops = circuit.num_operations();
    println!("Initial circuit has {} operations", initial_ops);

    // Method 1: Using tuple composition (recommended for known types)
    println!("\n--- Method 1: Tuple Composition ---");
    let composite_rewriter = (hadamard_rewriter.clone(), x_rewriter.clone());
    let optimiser = BadgerOptimiser::new(composite_rewriter, LexicographicCostFunction::default_cx_strategy());
    
    let optimised_tuple = optimiser.optimise(&circuit, Default::default());
    println!("After optimization with tuple composition: {} operations", optimised_tuple.num_operations());
    
    // Method 2: Using Vec<Box<dyn Rewriter<_>>> for dynamic composition
    println!("\n--- Method 2: Dynamic Composition ---");
    let rewriters: Vec<Box<dyn tket::rewrite::Rewriter<hugr::Node>>> = vec![
        Box::new(hadamard_rewriter),
        Box::new(x_rewriter),
    ];
    let optimiser_dynamic = BadgerOptimiser::new(rewriters, LexicographicCostFunction::default_cx_strategy());
    
    let optimised_dynamic = optimiser_dynamic.optimise(&circuit, Default::default());
    println!("After optimization with dynamic composition: {} operations", optimised_dynamic.num_operations());

    // Verify both methods produce equivalent results
    assert_eq!(optimised_tuple.num_operations(), optimised_dynamic.num_operations());
    
    // Only CX gates should remain (Hadamards and X gates cancelled out)
    assert_eq!(optimised_tuple.num_operations(), 2);
    assert!(optimised_tuple
        .operations()
        .all(|cmd| op_matches(cmd.optype(), TketOp::CX)));

    println!("\n✅ Success! Both composition methods work correctly.");
    println!("🎉 {} Hadamard pairs and {} X pairs were successfully cancelled!", 
             (initial_ops - 2) / 2 - 2,  // Account for CX gates
             4);  // 4 pairs of X gates
}
