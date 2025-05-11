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
  ]
}; 