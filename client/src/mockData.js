import { Brain, ListChecks } from 'lucide-react';

export const initialCourses = [
  {
    id: 'cs101',
    name: 'Probabilistic AI',
    description: 'Explore the fundamentals of probabilistic reasoning in AI.',
    progress: 73,
    topics: [
      { 
        id: 'topic1_1', 
        name: 'Introduction to Probability', 
        progress: 90, 
        files: [
          {name: 'Lecture1.pdf', url:'#'}, 
          {name:'Cheatsheet.png', url:'#'}
        ], 
        notes: 'Basic concepts of probability, sample spaces, events. Bayes\' theorem is crucial here.' 
      },
      { 
        id: 'topic1_2', 
        name: 'Bayesian Networks', 
        progress: 60, 
        files: [
          {name: 'Chapter2_BN.pdf', url:'#'}
        ], 
        notes: 'Understanding conditional independence and graphical models.' 
      },
      { 
        id: 'topic1_3', 
        name: 'Hidden Markov Models', 
        progress: 45, 
        files: [], 
        notes: 'Focus on sequences and state transitions.' 
      },
    ],
    color: 'bg-blue-500',
    icon: <Brain className="w-8 h-8 text-blue-300" />
  },
  {
    id: 'math202',
    name: 'Linear Algebra',
    description: 'Master vectors, matrices, and linear transformations.',
    progress: 45,
    topics: [
      { 
        id: 'topic2_1', 
        name: 'Vectors and Spaces', 
        progress: 70, 
        files: [], 
        notes: 'Vector addition, scalar multiplication, dot product.' 
      },
      { 
        id: 'topic2_2', 
        name: 'Matrix Operations', 
        progress: 30, 
        files: [], 
        notes: 'Matrix multiplication, determinants, inverses.' 
      },
    ],
    color: 'bg-green-500',
    icon: <ListChecks className="w-8 h-8 text-green-300" />
  },
];

export const mockQuizQuestions = {
  cs101_topic1_1: [
    { 
      id: 'q1', 
      text: 'What is the definition of conditional probability?', 
      type: 'recall', 
      explanation: 'Conditional probability P(A|B) is the likelihood of event A occurring given that B is true. Formula: P(A|B) = P(A ∩ B) / P(B). It is foundational for Bayes\' Theorem.' 
    },
    { 
      id: 'q2', 
      text: 'Explain Bayes\' Theorem and its components.', 
      type: 'test', 
      explanation: 'Bayes\' Theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event. Formula: P(A|B) = [P(B|A) * P(A)] / P(B).' 
    },
    {
      id: 'q3',
      text: 'What is the difference between joint probability and marginal probability?',
      type: 'test',
      explanation: 'Joint probability P(A,B) is the probability of both events A and B occurring together. Marginal probability P(A) is the probability of event A occurring regardless of event B. Marginal probability can be obtained by summing joint probabilities over all possible values of the other variable.'
    },
    {
      id: 'q4',
      text: 'Explain the concept of independence in probability theory.',
      type: 'test',
      explanation: 'Two events A and B are independent if the occurrence of one does not affect the probability of the other. Mathematically, P(A|B) = P(A) and P(B|A) = P(B). For independent events, P(A ∩ B) = P(A) * P(B).'
    }
  ],
  cs101_topic1_2: [
    { 
      id: 'q5', 
      text: 'What is a d-separation in Bayesian Networks?', 
      type: 'test', 
      explanation: 'D-separation (direction-dependent separation) is a graphical criterion used to determine whether a set of nodes X is independent of another set Y, given a third set Z in a Bayesian network.' 
    },
    {
      id: 'q6',
      text: 'Explain the concept of conditional independence in Bayesian Networks.',
      type: 'test',
      explanation: 'Conditional independence in Bayesian Networks means that two variables are independent given the values of their parents. This is a key property that allows for efficient computation of probabilities in the network.'
    },
    {
      id: 'q7',
      text: 'What is the Markov blanket of a node in a Bayesian Network?',
      type: 'recall',
      explanation: 'The Markov blanket of a node consists of its parents, children, and children\'s parents (spouses). It contains all the variables that shield the node from the rest of the network, making the node conditionally independent of all other variables given its Markov blanket.'
    },
    {
      id: 'q8',
      text: 'How do you perform inference in a Bayesian Network?',
      type: 'test',
      explanation: 'Inference in Bayesian Networks can be done through various methods: exact inference (variable elimination, junction tree algorithm) or approximate inference (sampling methods like Gibbs sampling, variational methods). The choice depends on the network structure and computational requirements.'
    }
  ],
  cs101_topic1_3: [
    {
      id: 'q9',
      text: 'What are the key components of a Hidden Markov Model?',
      type: 'recall',
      explanation: 'An HMM consists of: 1) Hidden states (unobservable), 2) Observable outputs, 3) Transition probabilities between states, 4) Emission probabilities (probabilities of observations given states), and 5) Initial state probabilities.'
    },
    {
      id: 'q10',
      text: 'Explain the Forward-Backward algorithm in HMMs.',
      type: 'test',
      explanation: 'The Forward-Backward algorithm computes the probability of being in a particular state at a given time, given the sequence of observations. It combines forward probabilities (probability of observations up to time t and being in state i) with backward probabilities (probability of future observations given current state).'
    },
    {
      id: 'q11',
      text: 'What is the Viterbi algorithm and when is it used?',
      type: 'test',
      explanation: 'The Viterbi algorithm finds the most likely sequence of hidden states given a sequence of observations. It uses dynamic programming to efficiently compute the optimal path through the state space. It\'s commonly used in speech recognition, natural language processing, and bioinformatics.'
    },
    {
      id: 'q12',
      text: 'How do you train a Hidden Markov Model?',
      type: 'test',
      explanation: 'HMMs are typically trained using the Baum-Welch algorithm (a special case of the EM algorithm). It iteratively updates the model parameters (transition probabilities, emission probabilities, and initial state probabilities) to maximize the likelihood of the observed data.'
    }
  ]
}; 