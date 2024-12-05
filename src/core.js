// Matrix operations
class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(rows)
      .fill()
      .map(() => Array(cols).fill(0));
  }

  static fromArray(arr) {
    const m = new Matrix(arr.length, 1);
    m.data = arr.map((x) => [x]);
    return m;
  }

  static subtract(a, b) {
    const result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return result;
  }

  toArray() {
    const arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  randomize() {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = Math.random() * 2 - 1;
      }
    }
  }

  add(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += n;
        }
      }
    }
  }

  static transpose(matrix) {
    const result = new Matrix(matrix.cols, matrix.rows);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }

  static multiply(a, b) {
    if (a.cols !== b.rows) {
      console.error("Columns of A must match rows of B.");
      return undefined;
    }
    const result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  multiply(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] *= n.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] *= n;
        }
      }
    }
  }

  map(func) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        const val = this.data[i][j];
        this.data[i][j] = func(val);
      }
    }
  }

  static map(matrix, func) {
    const result = new Matrix(matrix.rows, matrix.cols);
    for (let i = 0; i < matrix.rows; i++) {
      for (let j = 0; j < matrix.cols; j++) {
        const val = matrix.data[i][j];
        result.data[i][j] = func(val);
      }
    }
    return result;
  }

  copy() {
    const m = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        m.data[i][j] = this.data[i][j];
      }
    }
    return m;
  }
}

// Neural Network Layer with proper backpropagation
class Layer {
  constructor(inputSize, outputSize, activation = "relu") {
    this.weights = new Matrix(outputSize, inputSize);
    this.weights.randomize();
    this.weights.multiply(Math.sqrt(2.0 / inputSize)); // He initialization

    this.bias = new Matrix(outputSize, 1);
    this.bias.randomize();

    this.activation = activation;
    this.lastInput = null;
    this.lastOutput = null;
    this.lastActivation = null;
  }

  forward(input) {
    this.lastInput = input;

    // Weight multiplication and bias addition
    const output = Matrix.multiply(this.weights, input);
    output.add(this.bias);
    this.lastOutput = output;

    // Activation
    let activated;
    if (this.activation === "relu") {
      activated = Matrix.map(output, (x) => Math.max(0, x));
    } else if (this.activation === "softmax") {
      const expValues = Matrix.map(output, Math.exp);
      const sumExp = expValues.toArray().reduce((a, b) => a + b, 0);
      activated = Matrix.map(expValues, (x) => x / sumExp);
    } else {
      activated = output;
    }
    this.lastActivation = activated;
    return activated;
  }

  backward(outputGradient, learningRate) {
    let activationGradient;

    if (this.activation === "relu") {
      activationGradient = Matrix.map(this.lastOutput, (x) => (x > 0 ? 1 : 0));
      outputGradient = Matrix.multiply(outputGradient, activationGradient);
    } else if (this.activation === "softmax") {
      // Softmax gradient is handled in loss calculation
      activationGradient = outputGradient;
    }

    // Calculate gradients
    const weightGradient = Matrix.multiply(
      outputGradient,
      Matrix.transpose(this.lastInput)
    );
    const inputGradient = Matrix.multiply(
      Matrix.transpose(this.weights),
      outputGradient
    );

    // Update weights and biases
    this.weights.add(Matrix.map(weightGradient, (x) => x * learningRate));
    this.bias.add(Matrix.map(outputGradient, (x) => x * learningRate));

    return inputGradient;
  }
}

// Experience Replay Buffer
class ExperienceBuffer {
  constructor(maxSize) {
    this.maxSize = maxSize;
    this.buffer = [];
  }

  add(experience) {
    this.buffer.push(experience);
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift();
    }
  }

  sample(batchSize) {
    const indices = Array(batchSize)
      .fill()
      .map(() => Math.floor(Math.random() * this.buffer.length));
    return indices.map((i) => this.buffer[i]);
  }
}

// Policy Network with proper backpropagation
class PolicyNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    this.layer1 = new Layer(inputSize, hiddenSize, "relu");
    this.layer2 = new Layer(hiddenSize, outputSize, "softmax");
    this.experienceBuffer = new ExperienceBuffer(10000);
  }

  forward(state) {
    const input = Matrix.fromArray(state);
    const hidden = this.layer1.forward(input);
    const output = this.layer2.forward(hidden);
    return output.toArray();
  }

  sampleAction(state) {
    const probs = this.forward(state);
    let sum = 0;
    const r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      sum += probs[i];
      if (r <= sum) return i;
    }
    return probs.length - 1;
  }

  calculatePolicyGradient(states, actions, advantages) {
    const batchSize = states.length;
    let totalLoss = 0;

    // Forward pass and accumulate gradients
    const stateGrads = [];
    const actionGrads = [];

    for (let i = 0; i < batchSize; i++) {
      const state = Matrix.fromArray(states[i]);
      const action = actions[i];
      const advantage = advantages[i];

      // Forward pass
      const hidden = this.layer1.forward(state);
      const probs = this.layer2.forward(hidden);

      // Calculate loss and gradients
      const logProbs = Matrix.map(probs, (x) => Math.log(Math.max(x, 1e-7)));
      const actionLogProb = logProbs.data[action][0];
      totalLoss -= actionLogProb * advantage;

      // Policy gradient
      const gradients = new Matrix(probs.rows, probs.cols);
      gradients.data[action][0] = -advantage / probs.data[action][0];

      stateGrads.push(state);
      actionGrads.push(gradients);
    }

    return {
      loss: totalLoss / batchSize,
      stateGrads,
      actionGrads,
    };
  }

  update(states, actions, advantages, learningRate) {
    const { loss, stateGrads, actionGrads } = this.calculatePolicyGradient(
      states,
      actions,
      advantages
    );

    // Backward pass with accumulated gradients
    for (let i = 0; i < stateGrads.length; i++) {
      const outputGrad = actionGrads[i];
      const hiddenGrad = this.layer2.backward(outputGrad, learningRate);
      this.layer1.backward(hiddenGrad, learningRate);
    }

    return loss;
  }

  addExperience(state, action, reward, nextState, done) {
    this.experienceBuffer.add({ state, action, reward, nextState, done });
  }
}

// Complex Environment
class Environment {
  constructor() {
    this.reset();
  }

  reset() {
    // Physical state
    this.position = 0;
    this.velocity = 0;
    this.angle = 0;
    this.angularVelocity = 0;

    // Task parameters
    this.target = Math.random() * 2 - 1;
    this.energy = 1.0;
    this.time = 0;

    // Constants
    this.dt = 0.05; // Time step
    this.gravity = 9.81;
    this.mass = 1.0;
    this.length = 1.0;
    this.damping = 0.1;

    return this.getState();
  }

  getState() {
    return [
      this.position / 2,
      this.velocity / 5,
      Math.sin(this.angle),
      Math.cos(this.angle),
      this.angularVelocity / 5,
      (this.target - this.position) / 2,
      this.energy,
      Math.sin(this.time * 2 * Math.PI), // Periodic feature
    ];
  }

  step(action) {
    // Action processing: -1 to 1 range
    const force = (action - 1) * 10.0;

    // Physics simulation with RK4 integration
    const k1 = this.derivatives(
      this.position,
      this.velocity,
      this.angle,
      this.angularVelocity,
      force
    );
    const k2 = this.derivatives(
      this.position + (k1.dx * this.dt) / 2,
      this.velocity + (k1.dv * this.dt) / 2,
      this.angle + (k1.dtheta * this.dt) / 2,
      this.angularVelocity + (k1.dw * this.dt) / 2,
      force
    );
    const k3 = this.derivatives(
      this.position + (k2.dx * this.dt) / 2,
      this.velocity + (k2.dv * this.dt) / 2,
      this.angle + (k2.dtheta * this.dt) / 2,
      this.angularVelocity + (k2.dw * this.dt) / 2,
      force
    );
    const k4 = this.derivatives(
      this.position + k3.dx * this.dt,
      this.velocity + k3.dv * this.dt,
      this.angle + k3.dtheta * this.dt,
      this.angularVelocity + k3.dw * this.dt,
      force
    );

    // Update state
    this.position += ((k1.dx + 2 * k2.dx + 2 * k3.dx + k4.dx) * this.dt) / 6;
    this.velocity += ((k1.dv + 2 * k2.dv + 2 * k3.dv + k4.dv) * this.dt) / 6;
    this.angle +=
      ((k1.dtheta + 2 * k2.dtheta + 2 * k3.dtheta + k4.dtheta) * this.dt) / 6;
    this.angularVelocity +=
      ((k1.dw + 2 * k2.dw + 2 * k3.dw + k4.dw) * this.dt) / 6;

    // Energy consumption
    const energyCost = Math.abs(force) * this.dt * 0.1;
    this.energy = Math.max(0, this.energy - energyCost);

    this.time += this.dt;

    // Reward calculation
    const distanceToTarget = Math.abs(this.position - this.target);
    const positionReward = -distanceToTarget;
    const velocityPenalty = -Math.abs(this.velocity) * 0.1;
    const angularPenalty = -Math.abs(this.angularVelocity) * 0.1;
    const energyBonus = this.energy * 0.1;

    const reward =
      positionReward + velocityPenalty + angularPenalty + energyBonus;

    // Terminal conditions
    const done =
      distanceToTarget < 0.1 &&
      Math.abs(this.velocity) < 0.1 &&
      Math.abs(this.angularVelocity) < 0.1;

    return {
      state: this.getState(),
      reward,
      done,
    };
  }

  derivatives(x, v, theta, w, force) {
    // Return derivatives for RK4 integration
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    // Equations of motion for an inverted pendulum on a cart
    const dx = v;
    const dtheta = w;

    const totalMass = this.mass * 2; // Cart + pendulum mass
    const numerator =
      force + this.mass * this.length * w * w * sinTheta - this.damping * v;
    const denominator = totalMass - this.mass * cosTheta * cosTheta;

    const dv = numerator / denominator;
    const dw = (this.gravity * sinTheta - cosTheta * dv) / this.length;

    return { dx, dv, dtheta, dw };
  }
}

// Advantage estimation with GAE (Generalized Advantage Estimation)
class GAE {
  constructor(gamma = 0.99, lambda = 0.95) {
    this.gamma = gamma;
    this.lambda = lambda;
  }

  estimate(rewards, values, dones) {
    const advantages = new Array(rewards.length).fill(0);
    let lastGAE = 0;

    for (let t = rewards.length - 1; t >= 0; t--) {
      const nextValue = t === rewards.length - 1 ? 0 : values[t + 1];
      const nextNonTerminal = t === rewards.length - 1 ? 0 : 1 - dones[t + 1];

      const delta =
        rewards[t] + this.gamma * nextValue * nextNonTerminal - values[t];
      lastGAE = delta + this.gamma * this.lambda * nextNonTerminal * lastGAE;
      advantages[t] = lastGAE;
    }

    return advantages;
  }
}

// Baseline (Value) Network
class ValueNetwork {
  constructor(inputSize, hiddenSize) {
    this.layer1 = new Layer(inputSize, hiddenSize, "relu");
    this.layer2 = new Layer(hiddenSize, 1);
  }

  forward(state) {
    const input = Matrix.fromArray(state);
    const hidden = this.layer1.forward(input);
    const output = this.layer2.forward(hidden);
    return output.toArray()[0];
  }

  update(states, returns, learningRate) {
    let totalLoss = 0;

    for (let i = 0; i < states.length; i++) {
      const state = Matrix.fromArray(states[i]);
      const target = returns[i];

      // Forward pass
      const hidden = this.layer1.forward(state);
      const predicted = this.layer2.forward(hidden);

      // MSE loss gradient
      const error = predicted.data[0][0] - target;
      totalLoss += error * error;

      const outputGrad = new Matrix(1, 1);
      outputGrad.data[0][0] = 2 * error;

      // Backward pass
      const hiddenGrad = this.layer2.backward(outputGrad, learningRate);
      this.layer1.backward(hiddenGrad, learningRate);
    }

    return totalLoss / states.length;
  }
}

// PPO-style ratio clipping
function clipPPORatio(ratio, epsilon = 0.2) {
  return Math.min(Math.max(ratio, 1 - epsilon), 1 + epsilon);
}

// Training setup for both CL and MCL
class Trainer {
  constructor(env, config = {}) {
    this.env = env;
    this.config = {
      batchSize: 64,
      numEpisodes: 1000,
      maxSteps: 200,
      policyLR: 0.0003,
      valueLR: 0.001,
      gamma: 0.99,
      lambda: 0.95,
      entropyCoef: 0.01,
      ppoEpsilon: 0.2,
      ...config,
    };

    const stateSize = env.getState().length;
    const numActions = 3; // Left, Stay, Right

    // Initialize networks
    this.clPolicy = new PolicyNetwork(stateSize, 64, numActions);
    this.mclPolicy = new PolicyNetwork(stateSize, 64, numActions);
    this.clValue = new ValueNetwork(stateSize, 64);
    this.mclValue = new ValueNetwork(stateSize, 64);

    // Initialize GAE calculator
    this.gae = new GAE(this.config.gamma, this.config.lambda);

    // Statistics tracking
    this.clStats = { episodes: [], rewards: [], losses: [] };
    this.mclStats = { episodes: [], rewards: [], losses: [] };
  }

  async train() {
    for (let episode = 0; episode < this.config.numEpisodes; episode++) {
      // Run episode for both policies
      const clResult = await this.runEpisode(
        this.clPolicy,
        this.clValue,
        false
      );
      const mclResult = await this.runEpisode(
        this.mclPolicy,
        this.mclValue,
        true
      );

      // Update statistics
      this.clStats.episodes.push(episode);
      this.clStats.rewards.push(clResult.totalReward);
      this.clStats.losses.push(clResult.loss);

      this.mclStats.episodes.push(episode);
      this.mclStats.rewards.push(mclResult.totalReward);
      this.mclStats.losses.push(mclResult.loss);
    }

    return {
      clStats: this.clStats,
      mclStats: this.mclStats,
    };
  }

  async runEpisode(policy, value, isMCL = false) {
    let state = this.env.reset();
    let totalReward = 0;
    let done = false;

    const trajectory = {
      states: [],
      actions: [],
      rewards: [],
      values: [],
      logprobs: [],
      dones: [],
    };

    // Collect trajectory
    for (let step = 0; step < this.config.maxSteps && !done; step++) {
      const probs = policy.forward(state);
      const action = policy.sampleAction(state);
      const valueEst = value.forward(state);

      const { state: nextState, reward, done: isDone } = this.env.step(action);

      trajectory.states.push(state);
      trajectory.actions.push(action);
      trajectory.rewards.push(reward);
      trajectory.values.push(valueEst);
      trajectory.logprobs.push(Math.log(probs[action]));
      trajectory.dones.push(isDone);

      totalReward += reward;
      state = nextState;
      done = isDone;
    }

    // Calculate advantages and returns
    const advantages = this.gae.estimate(
      trajectory.rewards,
      trajectory.values,
      trajectory.dones
    );

    const returns = advantages.map((adv, i) => adv + trajectory.values[i]);

    // Normalize advantages
    const advMean = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const advStd = Math.sqrt(
      advantages.reduce((a, b) => a + Math.pow(b - advMean, 2), 0) /
        advantages.length
    );
    const normalizedAdvantages = advantages.map(
      (a) => (a - advMean) / (advStd + 1e-8)
    );

    // Policy update
    let policyLoss;
    if (isMCL) {
      // Maynard-Cross Learning update
      const valueEstimate =
        trajectory.values.reduce((a, b) => a + b, 0) / trajectory.values.length;
      policyLoss = policy.update(
        trajectory.states,
        trajectory.actions,
        normalizedAdvantages.map(
          (adv) => adv / Math.max(0.1, Math.abs(valueEstimate))
        ),
        this.config.policyLR
      );
    } else {
      // Standard policy update
      policyLoss = policy.update(
        trajectory.states,
        trajectory.actions,
        normalizedAdvantages,
        this.config.policyLR
      );
    }

    // Value network update
    const valueLoss = value.update(
      trajectory.states,
      returns,
      this.config.valueLR
    );

    return {
      totalReward,
      loss: policyLoss + valueLoss,
      averageValue:
        trajectory.values.reduce((a, b) => a + b, 0) / trajectory.values.length,
    };
  }
}

