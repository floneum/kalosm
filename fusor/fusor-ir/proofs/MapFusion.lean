import Std

universe u

structure Semantics (Value : Type u) where
  unary : Nat → Value → Value
  binary : Nat → Value → Value → Value
  ternary : Nat → Value → Value → Value → Value

inductive Expr (Value : Type u) (arity : Nat) where
  | arg : Fin arity → Expr Value arity
  | literal : Value → Expr Value arity
  | uniform : Nat → Expr Value arity
  | indexOf : Nat → Expr Value arity
  | unary : Nat → Expr Value arity → Expr Value arity
  | binary : Nat → Expr Value arity → Expr Value arity → Expr Value arity
  | ternary : Nat → Expr Value arity → Expr Value arity → Expr Value arity → Expr Value arity

def eval {Value : Type u} {arity : Nat} (sem : Semantics Value)
    (operands : Fin arity → Value) (uniforms coordinates : Nat → Value) : Expr Value arity → Value
  | .arg i => operands i
  | .literal value => value
  | .uniform symbol => uniforms symbol
  | .indexOf axis => coordinates axis
  | .unary op child => sem.unary op (eval sem operands uniforms coordinates child)
  | .binary op left right =>
      sem.binary op (eval sem operands uniforms coordinates left)
        (eval sem operands uniforms coordinates right)
  | .ternary op first second third =>
      sem.ternary op (eval sem operands uniforms coordinates first)
        (eval sem operands uniforms coordinates second)
        (eval sem operands uniforms coordinates third)

def substExpr {Value : Type u} {n m : Nat} (e : Expr Value n)
    (σ : Fin n → Expr Value m) : Expr Value m :=
  match e with
  | .arg i => σ i
  | .literal value => .literal value
  | .uniform symbol => .uniform symbol
  | .indexOf axis => .indexOf axis
  | .unary op child => .unary op (substExpr child σ)
  | .binary op left right => .binary op (substExpr left σ) (substExpr right σ)
  | .ternary op first second third =>
      .ternary op (substExpr first σ) (substExpr second σ) (substExpr third σ)

theorem eval_subst {Value : Type u} {n m : Nat} (e : Expr Value n)
    (σ : Fin n → Expr Value m) (sem : Semantics Value) (operands : Fin m → Value)
    (uniforms coordinates : Nat → Value) :
    eval sem operands uniforms coordinates (substExpr e σ) =
      eval sem (fun i => eval sem operands uniforms coordinates (σ i)) uniforms coordinates e := by
  induction e with
  | arg i => rfl
  | literal value => rfl
  | uniform symbol => rfl
  | indexOf axis => rfl
  | unary op child ih => simp [substExpr, eval, ih]
  | binary op left right ihLeft ihRight => simp [substExpr, eval, ihLeft, ihRight]
  | ternary op first second third ihFirst ihSecond ihThird =>
      simp [substExpr, eval, ihFirst, ihSecond, ihThird]

def shiftedIndex (n : Nat) {m : Nat} (i : Fin m) : Fin (n + m) :=
  ⟨n + i.val, by have hi := i.isLt; omega⟩

def frontIndex {n m : Nat} (i : Fin n) : Fin (n + m) :=
  ⟨i.val, by have hi := i.isLt; omega⟩

def retainedSlot {n : Nat} (k j : Fin (n + 1)) (h : j ≠ k) : Fin n :=
  if hlt : j.val < k.val then
    ⟨j.val, by have hk := k.isLt; omega⟩
  else
    ⟨j.val - 1, by
      have hj := j.isLt
      have hk := k.isLt
      have hneq : j.val ≠ k.val := by
        intro heq
        exact h (Fin.ext heq)
      omega⟩

def insertSlot {n : Nat} (k : Fin (n + 1)) (i : Fin n) : Fin (n + 1) :=
  if hlt : i.val < k.val then
    ⟨i.val, by have hi := i.isLt; omega⟩
  else
    ⟨i.val + 1, by have hi := i.isLt; have hk := k.isLt; omega⟩

theorem insertSlot_retained {n : Nat} (k j : Fin (n + 1)) (h : j ≠ k) :
    insertSlot k (retainedSlot k j h) = j := by
  apply Fin.ext
  by_cases hlt : j.val < k.val
  · simp [insertSlot, retainedSlot, hlt]
  · have hneq : j.val ≠ k.val := by
      intro heq
      exact h (Fin.ext heq)
    have hgt : k.val < j.val := by omega
    have hinsert : ¬ j.val - 1 < k.val := by omega
    simp [insertSlot, retainedSlot, hlt, hinsert]
    omega

def fusedOperands {Value : Type u} {n m : Nat} (consumer : Fin n → Value)
    (producerInputs : Fin m → Value) : Fin (n + m) → Value := by
  intro i
  exact if hlt : i.val < n then consumer ⟨i.val, hlt⟩ else
    producerInputs ⟨i.val - n, by have hi := i.isLt; omega⟩

theorem fusedOperands_front {Value : Type u} {n m : Nat} (consumer : Fin n → Value)
    (producerInputs : Fin m → Value) (i : Fin n) :
    fusedOperands consumer producerInputs (frontIndex i) = consumer i := by
  unfold fusedOperands frontIndex
  simp only [dif_pos i.isLt]

theorem fusedOperands_shifted {Value : Type u} {n m : Nat} (consumer : Fin n → Value)
    (producerInputs : Fin m → Value) (i : Fin m) :
    fusedOperands consumer producerInputs (shiftedIndex n i) = producerInputs i := by
  unfold fusedOperands shiftedIndex
  have hnot : ¬ n + i.val < n := by omega
  have hi := i.isLt
  simp only [dif_neg hnot]
  apply congrArg producerInputs
  apply Fin.ext
  change n + i.val - n = i.val
  omega

def shiftProducer {Value : Type u} {n m : Nat} (e : Expr Value m) : Expr Value (n + m) :=
  substExpr e (fun i => .arg (shiftedIndex n i))

def retainedIndex {n m : Nat} (k j : Fin (n + 1)) (h : j ≠ k) : Fin (n + m) :=
  frontIndex (m := m) (retainedSlot k j h)

theorem fusedOperands_retained {Value : Type u} {n m : Nat} (consumer : Fin n → Value)
    (producerInputs : Fin m → Value) (k j : Fin (n + 1)) (h : j ≠ k) :
    fusedOperands consumer producerInputs (retainedIndex k j h) = consumer (retainedSlot k j h) := by
  exact fusedOperands_front consumer producerInputs (retainedSlot k j h)

def spliceSubstitution {Value : Type u} {n m : Nat} (k : Fin (n + 1))
    (producer : Expr Value m) : Fin (n + 1) → Expr Value (n + m) := fun j =>
  if h : j = k then shiftProducer producer else .arg (retainedIndex k j h)

def originalOperands {Value : Type u} {n : Nat} (k : Fin (n + 1))
    (consumer : Fin (n + 1) → Value) (producerValue : Value) : Fin (n + 1) → Value := fun j =>
  if j = k then producerValue else consumer j

def compressedConsumer {Value : Type u} {n : Nat} (k : Fin (n + 1))
    (consumer : Fin (n + 1) → Value) : Fin n → Value := fun i => consumer (insertSlot k i)

theorem eval_shiftProducer {Value : Type u} {n m : Nat} (sem : Semantics Value)
    (consumer : Fin n → Value) (producerInputs : Fin m → Value)
    (uniforms coordinates : Nat → Value) (e : Expr Value m) :
    eval sem (fusedOperands consumer producerInputs) uniforms coordinates (shiftProducer e) =
      eval sem producerInputs uniforms coordinates e := by
  unfold shiftProducer
  rw [eval_subst]
  have hargs :
      (fun i => eval sem (fusedOperands consumer producerInputs) uniforms coordinates
        (.arg (shiftedIndex n i))) = producerInputs := by
    funext i
    exact fusedOperands_shifted consumer producerInputs i
  rw [hargs]

theorem spliceEnvironment {Value : Type u} {n m : Nat} (sem : Semantics Value)
    (consumer : Fin (n + 1) → Value) (producerInputs : Fin m → Value)
    (uniforms coordinates : Nat → Value) (k : Fin (n + 1)) (producer : Expr Value m) :
    (fun j => eval sem (fusedOperands (compressedConsumer k consumer) producerInputs)
      uniforms coordinates (spliceSubstitution k producer j)) =
    originalOperands k consumer (eval sem producerInputs uniforms coordinates producer) := by
  funext j
  by_cases h : j = k
  · subst j
    simpa [spliceSubstitution, originalOperands] using
      (eval_shiftProducer sem (compressedConsumer k consumer) producerInputs
        uniforms coordinates producer)
  · simp only [spliceSubstitution, originalOperands, dif_neg h, if_neg h, eval]
    rw [fusedOperands_retained]
    change consumer (insertSlot k (retainedSlot k j h)) = consumer j
    rw [insertSlot_retained]

theorem single_splice {Value : Type u} {n m : Nat} (sem : Semantics Value)
    (consumer : Fin (n + 1) → Value) (producerInputs : Fin m → Value)
    (uniforms coordinates : Nat → Value) (k : Fin (n + 1))
    (consumerBody : Expr Value (n + 1)) (producerBody : Expr Value m) :
    eval sem (fusedOperands (compressedConsumer k consumer) producerInputs)
        uniforms coordinates (substExpr consumerBody (spliceSubstitution k producerBody)) =
      eval sem (originalOperands k consumer
        (eval sem producerInputs uniforms coordinates producerBody))
        uniforms coordinates consumerBody := by
  rw [eval_subst]
  rw [spliceEnvironment sem consumer producerInputs uniforms coordinates k producerBody]

theorem map_fusion {Index : Type} {Value : Type u} {n m : Nat} (sem : Semantics Value)
    (consumer : Index → Fin (n + 1) → Value) (producerInputs : Index → Fin m → Value)
    (uniforms coordinates : Index → Nat → Value) (k : Fin (n + 1))
    (consumerBody : Expr Value (n + 1)) (producerBody : Expr Value m) :
    (fun t => eval sem (fusedOperands (compressedConsumer k (consumer t)) (producerInputs t))
      (uniforms t) (coordinates t)
      (substExpr consumerBody (spliceSubstitution k producerBody))) =
    (fun t => eval sem (originalOperands k (consumer t)
      (eval sem (producerInputs t) (uniforms t) (coordinates t) producerBody))
      (uniforms t) (coordinates t) consumerBody) := by
  funext t
  exact single_splice sem (consumer t) (producerInputs t) (uniforms t) (coordinates t)
    k consumerBody producerBody

#print axioms eval_subst
#print axioms single_splice
#print axioms map_fusion
