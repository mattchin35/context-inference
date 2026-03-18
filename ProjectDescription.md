
# **Project Description: Mouse Behavior Analysis and Simulation**
A high-level description of what this codebase is for.

## **Overview**

This project analyzes mouse behavior and electrophysiology from a 2-choice contextual bandit task and uses simulations to:

1. Compare behavioral models (RL, HMM, hybrid models) to mouse data
2. Test whether GLM-HMM and LM-HMM can correctly recover underlying behavioral strategies

---

# **Task Description**

Mice perform a 2-choice task with hidden context:

* Two contexts: **Left** and **Right**
* Only the context-matching choice has nonzero reward probability
* Incorrect choice has reward probability = 0

### **Reward Structure**

* Correct choice reward probability: **p_reward = 0.6–0.9**
* Incorrect choice: **0**

### **Context Transitions**

* After any **correct choice** (rewarded or not), context switches with probability:

  * **p_switch = 0.2–0.3**
* Otherwise, context remains the same

### **Key Property: Asymmetric Evidence**

* **Reward → definitive evidence about context**
* **Omission → ambiguous (could be noise or switch)**

### **Optimal Strategy (Qualitative)**

* Stay after reward
* Switch after repeated omissions
* Be uncertain after a single omission

This task is a **POMDP / contextual bandit**.

---

# **Data Description**

* Neural recordings: **Neuropixels 2.0 (PFC + HPC)**
* Spike sorting: **Kilosort**
* Neural timestamps: **IRIG-H → UTC**
* Behavior: Raspberry Pi (GPS-synced)

Primary data:

* Trial-level behavior (choice, reward)
* Spike times (to be binned)

Future additions: pupil, treadmill, video

---

# **Behavioral Modeling**

## **Goal**

Compare mouse behavior to candidate strategies:

* Reinforcement learning (RL)
* Bayesian belief (HMM-style)
* Hybrid belief + decay + doubt models

All models may include **perseveration (choice stickiness)**.

---

## **GLM-HMM (Trial-Level Analysis)**

Predicts choice from trial history using bounded regressors.

### Regressor families:

* **Value-based:** Q-learning / forgetting Q-learning
* **Belief-based:** Bayesian log-odds (± tanh)
* **Omission/doubt:** accumulates across unrewarded trials
* **Hybrid belief+doubt:** reward-driven belief + omission-driven decay + doubt
* **History-based:** perseveration

### Model:

* Input: regressors at trial *t*
* Output: choice at trial *t+1*
* Produces:

  * Discrete state sequence
  * State-specific GLM weights

---

## **LM-HMM (Block-Level Analysis)**

Operates on **blocks between context switches**.

### Features:

* Total rewards in block
* Trials-to-switch after block change

### Interpretation:

* RL-like strategies → strong dependence on reward history
* Inference strategies → weak dependence

### Output:

* Block-level state assignments
* State-dependent regression weights

---

## **Model Selection**

* Number of states: **1–5**
* Selected via **BIC + cross-validation**
* Implement using **ssm only (no reimplementation)**

---

# **Behavior Simulation**

## **Goal**

Generate synthetic behavior to:

* Compare against mouse data
* Test GLM-HMM / LM-HMM recovery

---

## **Strategies**

* Forgetting Q-learning
* Bayesian belief (no decay)
* Bayesian belief + omission decay
* Hybrid belief + decay + doubt
* Any of the above + perseveration

Parameters should enforce **multi-trial evidence accumulation** (no instant switching).

---

## **Strategy Switching**

Max **3 strategies per session**

Three switching regimes:

### 1. Independent States

* Only active strategy updates
* Others frozen

### 2. Partially Coupled States

* Active strategy updates fully
* Inactive strategies update weakly (decay / hazard)

### 3. Shared State

* Single latent state
* Strategy determines update rule

---

## **Simulation Design**

* Choices generated probabilistically (logistic/softmax)
* Default switching: **blocks of ~50 trials**
* Optional: reproducible via random seed

Note: inferred number of HMM states may differ from true number.

---

# **Model Evaluation**

## **Recovery Metrics (Simulation)**

* State accuracy vs ground truth
* Alignment of inferred vs true switches
* Match between inferred and true strategy groupings
* Recovery of regressor weights

## **Failure Modes**

* State merging
* State splitting
* Misaligned transitions

---

# **Neural Data Integration**

* Align behavior and spikes via **IRIG → UTC**
* Store in **pynapple-compatible format**

## **Analysis Goals**

* Encoding of:

  * context, choice, reward
  * belief / uncertainty signals
* PFC–HPC communication
* Trial and across-trial dynamics

## **Approaches (examples)**

* PCA / LDA / gcPCA
* GLMs (neural ↔ behavior, neural ↔ neural)
* CCA / communication subspaces
* Latent variable models (SCA, LFADS)

---

# **Code Frameworks**

* Core: numpy, scipy, pandas, sklearn
* Stats: statsmodels
* Neuroscience: pynapple, ssm
* Plotting: matplotlib, seaborn

---

# **Future Directions (Optional)**

* Continuous or infinite-state HMMs
* RNN modeling (behavior or neural)
* LFP analyses
* Generative neural simulations

---

# **Key References**

(To be populated)


