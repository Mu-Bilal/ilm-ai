import { Brain, ListChecks, BookOpen } from 'lucide-react';

export const initialCourses = [
  {
    id: 'cs101',
    name: 'Probabilistic AI',
    description: 'A comprehensive course on probabilistic methods in artificial intelligence, covering Bayesian learning, Gaussian processes, and reinforcement learning.',
    progress: 75,
    topics: [
      {
        id: '1 Fundamentals',
        name: '1 Fundamentals',
        progress: 85,
        notes: 'Introduction to probability theory and basic concepts.',
        files: []
      },
      {
        id: '2 Bayesian Linear Regression',
        name: '2 Bayesian Linear Regression',
        progress: 65,
        notes: 'Understanding Bayesian approaches to linear regression.',
        files: []
      },
      {
        id: '3 Kalman Filters',
        name: '3 Kalman Filters',
        progress: 75,
        notes: 'Study of Kalman filters and their applications.',
        files: []
      },
      {
        id: '4 Gaussian Processes',
        name: '4 Gaussian Processes',
        progress: 70,
        notes: 'Introduction to Gaussian processes and their use in machine learning.',
        files: []
      },
      {
        id: '5 Variational Inference',
        name: '5 Variational Inference',
        progress: 60,
        notes: 'Methods for approximate Bayesian inference.',
        files: []
      },
      {
        id: '6 Markov Chain Monte Carlo Methods',
        name: '6 Markov Chain Monte Carlo Methods',
        progress: 55,
        notes: 'Sampling methods for complex probability distributions.',
        files: []
      },
      {
        id: '7 Bayesian Deep Learning',
        name: '7 Bayesian Deep Learning',
        progress: 50,
        notes: 'Bayesian approaches to deep neural networks.',
        files: []
      },
      {
        id: '8 Active Learning',
        name: '8 Active Learning',
        progress: 65,
        notes: 'Methods for efficient data collection and learning.',
        files: []
      },
      {
        id: '9 Bayesian Optimization',
        name: '9 Bayesian Optimization',
        progress: 70,
        notes: 'Optimization techniques using Bayesian methods.',
        files: []
      },
      {
        id: '10 Markov Decision Processes',
        name: '10 Markov Decision Processes',
        progress: 60,
        notes: 'Framework for modeling decision making in uncertain environments.',
        files: []
      },
      {
        id: '11 Tabular Reinforcement Learning',
        name: '11 Tabular Reinforcement Learning',
        progress: 55,
        notes: 'Basic reinforcement learning algorithms for discrete state spaces.',
        files: []
      },
      {
        id: '12 Model-free Approximate Reinforcement Learning',
        name: '12 Model-free Approximate Reinforcement Learning',
        progress: 50,
        notes: 'Advanced reinforcement learning without explicit models.',
        files: []
      },
      {
        id: '13 Model-based Approximate Reinforcement Learning',
        name: '13 Model-based Approximate Reinforcement Learning',
        progress: 45,
        notes: 'Reinforcement learning with learned environment models.',
        files: []
      }
    ],
    color: 'bg-purple-500',
    icon: <BookOpen className="w-8 h-8 text-purple-100" />
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
  cs101: [
    { 
      id: 'q1', 
      text: 'What is the definition of conditional probability?', 
      type: 'recall', 
      explanation: 'Conditional probability P(A|B) is the likelihood of event A occurring given that B is true. Formula: P(A ∩ B) / P(B). It is foundational for Bayes\' Theorem.' 
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
    },
    {
      id: 'q5',
      text: 'Which of the following correctly describes the law of total probability?',
      type: 'test',
      questionType: 'multipleChoice',
      options: [
        'The probability of an event equals the sum of the probabilities of its possible outcomes',
        'The probability of an event can be calculated using all possible values of another random variable',
        'The probability of an event equals the product of conditional probabilities',
        'The probability of an event is always equal to 1 minus the probability of its complement'
      ],
      correctOptionIndex: 1,
      explanation: 'The law of total probability states that the probability of an event can be calculated by considering all possible values of another random variable. Mathematically, P(A) = Σ P(A|B_i)P(B_i) for a partition B_i of the sample space.'
    },
    {
      id: 'q6',
      text: 'Which approach is central to Bayesian statistics?',
      type: 'test',
      questionType: 'multipleChoice',
      options: [
        'Assuming all probabilities are equal',
        'Treating parameters as random variables with prior distributions',
        'Maximizing the likelihood function',
        'Using only frequentist methods'
      ],
      correctOptionIndex: 1,
      explanation: 'Bayesian statistics treats parameters as random variables with prior distributions, which are updated based on observed data to form posterior distributions. This is in contrast to frequentist statistics, which treats parameters as fixed but unknown values.'
    }
  ],
  'cs101_1 Fundamentals': [
    {
      id: 'q_fund_1',
      text: 'What are the three axioms of probability?',
      type: 'test',
      explanation: 'The three axioms of probability are: 1) Non-negativity: P(A) ≥ 0 for any event A, 2) Normalization: P(Ω) = 1 for the entire sample space Ω, and 3) Additivity: For mutually exclusive events A and B, P(A ∪ B) = P(A) + P(B).'
    },
    {
      id: 'q_fund_2',
      text: 'Which of these statements about random variables is true?',
      type: 'test',
      questionType: 'multipleChoice',
      options: [
        'Random variables can only take discrete values',
        'Continuous random variables have probability mass functions',
        'A random variable is a function that maps outcomes to real numbers',
        'Expected values can only be calculated for discrete random variables'
      ],
      correctOptionIndex: 2,
      explanation: 'A random variable is a function that maps outcomes from a sample space to real numbers. This allows us to apply mathematical operations to outcomes from random experiments.'
    },
    {
      id: 'q_fund_3',
      text: 'What does the expected value of a random variable represent?',
      type: 'recall',
      questionType: 'multipleChoice',
      options: [
        'The most likely outcome',
        'The middle value in the distribution',
        'The weighted average of all possible values',
        'The maximum possible value'
      ],
      correctOptionIndex: 2,
      explanation: 'The expected value (or mean) of a random variable is the weighted average of all possible values, where the weights are given by the probabilities. It represents the long-term average value if the random experiment is repeated many times.'
    }
  ]
}; 