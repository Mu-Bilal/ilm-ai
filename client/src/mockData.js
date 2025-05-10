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
    icon: <Brain className="w-8 h-8 text-blue-100" />
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
    icon: <ListChecks className="w-8 h-8 text-green-100" />
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
  ],
  cs101_topic1_2: [
    { 
      id: 'q3', 
      text: 'What is a d-separation in Bayesian Networks?', 
      type: 'test', 
      explanation: 'D-separation (direction-dependent separation) is a graphical criterion used to determine whether a set of nodes X is independent of another set Y, given a third set Z in a Bayesian network.' 
    },
  ]
}; 