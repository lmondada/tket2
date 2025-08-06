//! Advanced examples of composing multiple rewriters with Badger optimization.
//!
//! This example demonstrates various patterns for combining rewriters:
//! 1. Simple tuple composition
//! 2. Nested compositions
//! 3. Dynamic collections
//! 4. Mixed rewriter types

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
        MatchReplaceRewriter, Rewriter,
    },
    Circuit, TketOp,
};

/// Matcher for consecutive Hadamard gates
#[derive(Clone, Copy, Debug)]
struct HadamardPairMatcher;

/// Matcher for consecutive X gates  
#[derive(Clone, Copy, Debug)]
struct XPairMatcher;

/// Matcher for consecutive Z gates
#[derive(Clone, Copy, Debug)]
struct ZPairMatcher;

/// Simple gate cancellation replacement
#[derive(Clone, Copy, Debug)]
struct GateCancellation;

/// Match state tracking
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
enum TwoGateState {
    #[default]
    None,
    One,
}

impl CircuitMatcher for HadamardPairMatcher {
    type PartialMatchInfo = TwoGateState;
    type MatchInfo = ();

    fn match_tket_op(
        &self,
        op: TketOp,
        _op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        if op != TketOp::H {
            return MatchOutcome::stop();
        }

        match match_context.match_info {
            TwoGateState::None => MatchOutcome::default().proceed(TwoGateState::One),
            TwoGateState::One => MatchOutcome::default()
                .complete(())
                .skip(TwoGateState::One),
        }
    }
}

impl CircuitMatcher for XPairMatcher {
    type PartialMatchInfo = TwoGateState;
    type MatchInfo = ();

    fn match_tket_op(
        &self,
        op: TketOp,
        _op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        if op != TketOp::X {
            return MatchOutcome::stop();
        }

        match match_context.match_info {
            TwoGateState::None => MatchOutcome::default().proceed(TwoGateState::One),
            TwoGateState::One => MatchOutcome::default()
                .complete(())
                .skip(TwoGateState::One),
        }
    }
}

impl CircuitMatcher for ZPairMatcher {
    type PartialMatchInfo = TwoGateState;
    type MatchInfo = ();

    fn match_tket_op(
        &self,
        op: TketOp,
        _op_args: &[OpArg],
        match_context: MatchContext<Self::PartialMatchInfo, impl hugr::HugrView>,
    ) -> MatchOutcome<Self::PartialMatchInfo, Self::MatchInfo> {
        if op != TketOp::Z {
            return MatchOutcome::stop();
        }

        match match_context.match_info {
            TwoGateState::None => MatchOutcome::default().proceed(TwoGateState::One),
            TwoGateState::One => MatchOutcome::default()
                .complete(())
                .skip(TwoGateState::One),
        }
    }
}

impl MatchReplacement<()> for GateCancellation {
    fn replace_match<H: hugr::HugrView>(
        &self,
        subgraph: &SiblingSubgraph<H::Node>,
        _hugr: H,
        _match_info: (),
    ) -> Vec<Circuit> {
        assert_eq!(subgraph.node_count(), 2);
        
        // Return identity (empty circuit)
        let h = DFGBuilder::new(endo_sig(qb_t())).unwrap();
        let inps = h.input_wires();
        let empty_circ = h.finish_hugr_with_outputs(inps).unwrap();
        vec![Circuit::new(empty_circ)]
    }
}

/// Custom rewriter that does nothing (for demonstration)
#[derive(Clone, Debug)]
struct NoOpRewriter;

impl Rewriter<hugr::Node> for NoOpRewriter {
    fn get_rewrites(&self, _circ: &Circuit<impl hugr::HugrView<Node = hugr::Node>>) -> Vec<tket::rewrite::CircuitRewrite<hugr::Node>> {
        vec![] // No rewrites
    }
}

/// Create a test circuit with various gate patterns
fn create_complex_circuit() -> Circuit {
    let mut h = DFGBuilder::new(endo_sig(vec![qb_t(); 3])).unwrap();
    let qbs = h.input_wires();
    let mut circ = h.as_circuit(qbs);

    // Add patterns that should be optimized away
    // H-H pairs on qubit 0
    circ.append(TketOp::H, [0]).unwrap();
    circ.append(TketOp::H, [0]).unwrap();
    
    // X-X pairs on qubit 1
    circ.append(TketOp::X, [1]).unwrap();
    circ.append(TketOp::X, [1]).unwrap();
    
    // Z-Z pairs on qubit 2
    circ.append(TketOp::Z, [2]).unwrap();
    circ.append(TketOp::Z, [2]).unwrap();
    
    // Some CX gates that won't be optimized
    circ.append(TketOp::CX, [0, 1]).unwrap();
    circ.append(TketOp::CX, [1, 2]).unwrap();
    
    // More cancellable pairs
    circ.append(TketOp::H, [0]).unwrap();
    circ.append(TketOp::H, [0]).unwrap();

    let qbs = circ.finish();
    Circuit::new(h.finish_hugr_with_outputs(qbs).unwrap())
}

fn main() {
    println!("=== Advanced Rewriter Composition Examples ===\n");

    let circuit = create_complex_circuit();
    let initial_ops = circuit.num_operations();
    println!("Initial circuit has {} operations", initial_ops);
    
    // Create individual rewriters
    let h_rewriter = MatchReplaceRewriter::new(HadamardPairMatcher, GateCancellation);
    let x_rewriter = MatchReplaceRewriter::new(XPairMatcher, GateCancellation);
    let z_rewriter = MatchReplaceRewriter::new(ZPairMatcher, GateCancellation);
    let noop_rewriter = NoOpRewriter;

    println!("\n--- Example 1: Simple Tuple Composition ---");
    let simple_composite = (h_rewriter.clone(), x_rewriter.clone());
    let optimiser1 = BadgerOptimiser::new(simple_composite, LexicographicCostFunction::default_cx_strategy());
    let result1 = optimiser1.optimise(&circuit, Default::default());
    println!("H+X optimization: {} → {} operations", initial_ops, result1.num_operations());

    println!("\n--- Example 2: Triple Composition ---");
    let triple_composite = (h_rewriter.clone(), x_rewriter.clone(), z_rewriter.clone());
    let optimiser2 = BadgerOptimiser::new(triple_composite, LexicographicCostFunction::default_cx_strategy());
    let result2 = optimiser2.optimise(&circuit, Default::default());
    println!("H+X+Z optimization: {} → {} operations", initial_ops, result2.num_operations());

    println!("\n--- Example 3: Mixed Rewriter Types ---");
    let mixed_composite = (triple_composite, noop_rewriter.clone());
    let optimiser3 = BadgerOptimiser::new(mixed_composite, LexicographicCostFunction::default_cx_strategy());
    let result3 = optimiser3.optimise(&circuit, Default::default());
    println!("Mixed types: {} → {} operations", initial_ops, result3.num_operations());

    println!("\n--- Example 4: Dynamic Collection ---");
    let dynamic_rewriters: Vec<Box<dyn Rewriter<hugr::Node>>> = vec![
        Box::new(h_rewriter.clone()),
        Box::new(x_rewriter.clone()), 
        Box::new(z_rewriter.clone()),
        Box::new(noop_rewriter),
    ];
    let optimiser4 = BadgerOptimiser::new(dynamic_rewriters, LexicographicCostFunction::default_cx_strategy());
    let result4 = optimiser4.optimise(&circuit, Default::default());
    println!("Dynamic collection: {} → {} operations", initial_ops, result4.num_operations());

    println!("\n--- Example 5: Large Tuple (8 rewriters) ---");
    let large_composite = (
        h_rewriter.clone(), x_rewriter.clone(), z_rewriter.clone(), h_rewriter.clone(),
        x_rewriter.clone(), z_rewriter.clone(), h_rewriter, x_rewriter,
    );
    let optimiser5 = BadgerOptimiser::new(large_composite, LexicographicCostFunction::default_cx_strategy());
    let result5 = optimiser5.optimise(&circuit, Default::default());
    println!("8-tuple composition: {} → {} operations", initial_ops, result5.num_operations());

    // Verify consistency across different composition methods
    println!("\n--- Verification ---");
    assert_eq!(result2.num_operations(), result3.num_operations(), 
               "Mixed and triple composition should give same results");
    assert_eq!(result2.num_operations(), result4.num_operations(),
               "Triple and dynamic should give same results");
    
    // Only CX gates should remain (2 CX gates from the original circuit)
    assert_eq!(result2.num_operations(), 2);
    assert!(result2.operations().all(|cmd| op_matches(cmd.optype(), TketOp::CX)));
    
    println!("✅ All composition methods work correctly!");
    println!("🎯 Optimized away {} redundant gate pairs!", (initial_ops - 2) / 2);
}