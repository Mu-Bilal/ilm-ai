#### *Andreas Krause*
# Probabilistic Artificial Intelligence
###### Fall 2023

Institute for Machine Learning
Department of Computer Science

**Contents.** Bayesian learning and inference. Bayesian linear regression, Gaussian processes, Bayesian Deep Learning. Variational
inference, Markov chain Monte Carlo methods. Active learning, Bayesian optimization and the exploration-exploitation dilemma.
Markov decision processes, and Reinforcement learning in tabular and large state-action spaces with model-based and model-free
approaches.

acknowledgement

These notes were written and edited by **Jonas Hübotter** .

 - Many figures were inspired by the accompanying Jupyter notebooks created by Sebastian Curi.

 - Parts of section 1.1 were adapted from earlier notes by Mohammad Reza Karimi.

 - Figure 12.2 was contributed by Hado van Hasselt.

 - Some exercises are adapted from previous iterations of the course.

- We thank Zhiyuan Hu and Shyam Sundhar Ramesh for proofreading.

contributing

You are encouraged to raise issues and suggest fixes for anything you think can be improved.
[Contact: pai-script@lists.inf.ethz.ch](mailto:pai-script@lists.inf.ethz.ch)
Repository: `https://gitlab.inf.ethz.ch/OU-KRAUSE/pai-script`

jupyter notebooks with examples
```
https://gitlab.inf.ethz.ch/OU-KRAUSE/pai-demos

```
syllabus

1. *Introduction* chapter 1
2. *Bayesian Linear Regression & Filtering* chapter 2 through to section 2.4 and chapter 3
3. *Gaussian Processes 1* rest of chapter 2 and chapter 4 through to section 4.3
4. *Gaussian Processes 2* rest of chapter 4
5. *Variational Inference* chapter 5 up to section 5.6.1
6. *Markov Chain Monte Carlo* section 5.6.1 and chapter 6
7. *Bayesian Deep Learning* chapter 7
8. *Active Learning & Bayesian Optimization* chapter 8 and chapter 9
9. *Markov Decision Processes* chapter 10
10. *Reinforcement Learning 1* chapter 11
11. *Reinforcement Learning 2* chapter 12 through to section 12.3
12. *Reinforcement Learning 3* section 12.4
13. *Reinforcement Learning 4* chapter 13

Compilation date: **September 11, 2023**

This set of notes was written for the course Probabilistic Artificial Intelligence (263-5210-00L) at ETH Zürich.
Distribution of these notes without the permission of the authors is prohibited.

© 2023 ETH Zürich. All rights reserved.

### *Contents*
###### Preface vii Summary of Notation ix 1 Fundamentals 1

*1.1* 1
*Probability*
*1.2 Supervised Learning* 18
*1.3 Bayesian Learning and Inference* 19
*1.4 Parameter Estimation* 22
*1.5 Optimization* 30
*1.6 Multivariate Normal Distribution* 37
###### *I* Probabilistic Machine Learning 45 2 Bayesian Linear Regression 47

*2.1 Linear Regression* 47
*2.2 Uncertainty* 48
*2.3 Weight-space View* 50
*2.4 Recursive Bayesian Updates* 55
*2.5 Non-linear Regression* 56
*2.6 Function-space View* 57
###### 3 Kalman Filters 61

*3.1 Bayesian Filtering* 63
*3.2 Kalman Update* 65

iv

*3.3 Predicting* 66
###### 4 Gaussian Processes 69

*4.1 Learning and Inference* 70
*4.2 Sampling* 71
*4.3 Kernel Functions* 71
*4.4 Model Selection* 78
*4.5 Approximations* 83
###### 5 Variational Inference 91

*5.1 Laplace Approximation* 91
*5.2 Inference with a Variational Posterior* 95
*5.3 Information Theory* 98
*5.4 Variational Families* 103
*5.5 Kullback-Leibler Divergence* 104
*5.6 Evidence Lower Bound* 111
###### *6 Markov Chain Monte Carlo Methods* 119

*6.1 Markov Chains* 120

*6.2 Elementary Sampling Methods* 128
*6.3 Langevin Dynamics* 131
*6.4 Hamiltonian Monte Carlo* 136
###### 7 Bayesian Deep Learning 139

*7.1 Artificial Neural Networks* 139
*7.2 Bayesian Neural Networks* 143
*7.3 Maximum A Posteriori Inference* 143
*7.4 Approximate Inference* 144
*7.5 Calibration* 149
###### II Sequential Decision Making 153 *8* Active Learning 155

*8.1 Conditional Entropy* 155
*8.2 Mutual Information* 157

v


*8.3 Submodularity of Mutual Information* 160
*8.4 Maximizing Mutual Information* 162
###### 9 Bayesian Optimization 167

*9.1 Exploration-Exploitation Dilemma* 167
*9.2 Online Learning and Bandits* 168
*9.3 Acquisition Functions* 170
*9.4 Model Selection* 176
###### *10 Markov Decision Processes* 179

*10.1 Bellman Expectation Equation* 182
*10.2 Policy Evaluation* 183
*10.3 Policy Optimization* 185
*10.4 Partial Observability* 193
###### 11 Tabular Reinforcement Learning 199

*11.1 The Reinforcement Learning Problem* 199
*11.2 Model-based Approaches* 201
*11.3 Model-free Approaches* 206
###### 12 Model-free Approximate Reinforcement Learning 215

*12.1 Tabular Reinforcement Learning as Optimization* 215
*12.2 Value Function Approximation* 217
*12.3 Policy Approximation* 222
*12.4 Actor-Critic Methods* 229
###### 13 Model-based Approximate Reinforcement Learning 245

*13.1 Planning* 246
*13.2 Learning* 252
*13.3 Exploration* 258
###### A Solutions 269 Bibliography 311

vi
###### Acronyms 319 *Index* 321

### *Preface*

Artificial intelligence commonly refers to the science and engineering
of artificial systems that can carry out tasks generally associated with
requiring aspects of human intelligence, such as playing games, translating languages, and driving cars. In recent years, there have been
exciting advances in learning-based, data-driven approaches towards
AI, and machine learning and deep learning have enabled computer
systems to perceive the world in unprecedented ways. Reinforcement
learning has enabled breakthroughs in complex games such as Go and
challenging robotics tasks such as quadrupedal locomotion.

A key aspect of intelligence is to not only make predictions, but reason
about the *uncertainty* in these predictions, and to consider this uncertainty when making decisions. This is what the course “Probabilistic
Artificial Intelligence” is about. The first part covers probabilistic approaches to machine learning. We discuss how Bayesian approaches
to learning allow to differentiate between “epistemic” uncertainty due
to lack of data and “aleatoric” uncertainty, which is irreducible and
stems, e.g., from noisy observations and outcomes. We discuss concrete approaches towards Bayesian learning, such as Bayesian linear regression, Gaussian process models and Bayesian neural networks. Often, inference and making predictions with such models is intractable,
and we discuss modern approaches to efficient approximate inference.

The second part of the course is about taking uncertainty into account
in sequential decision tasks. We consider active learning and Bayesian
optimization — approaches that collect data by proposing experiments
that are informative for reducing the epistemic uncertainty. We then
consider reinforcement learning, a rich formalism for modeling agents
that learn to act in uncertain environments. After covering the basic
formalism of Markov Decision Processes, we consider modern deep
RL approaches that use neural network function approximation. We
close by discussing modern approaches in model-based RL, which harness epistemic and aleatoric uncertainty to guide exploration, while
also reasoning about safety.

### *Summary of Notation*

We follow these general rules:

- uppercase italic for constants *N*

- lowercase italic for indices *i* and scalar variables *x*

- lowercase italic bold for vectors ***x***, entries are denoted ***x*** ( *i* )

- uppercase italic bold for matrices ***M***, entries are denoted ***M*** ( *i*, *j* )

- uppercase italic for random variables *X*

- uppercase bold for random vectors **X**, entries are denoted **X** ( *i* )

- uppercase italic for sets *A*

- uppercase calligraphy for spaces (usually infinite sets) *A*

= .
equality by definition
*≈* approximately equals
∝ proportional to (up to multiplicative constants), *f* ∝ *g* iff *∃* *k* . *∀* *x* . *f* ( *x* ) = *k* *·* *g* ( *x* )
const an (additive) constant
**N** set of natural numbers *{* 1, 2, . . . *}*
**N** 0 set of natural numbers, including 0, **N** *∪{* 0 *}*

**R** set of real numbers

[ *m* ] set of natural numbers from 1 to *m*, *{* 1, 2, . . ., *m* *−* 1, *m* *}*
*i* : *j* subset of natural numbers between *i* and *j*, *{* *i*, *i* + 1, . . ., *j* *−* 1, *j* *}*
( *a*, *b* ] real interval between *a* and *b* including *b* but not including *a*
*f* : *A* *→* *B* function *f* from elements of set *A* to elements of set *B*
*f* *◦* *g* function composition, *f* ( *g* ( *·* ))
( *·* ) + max *{* 0, *·}*
log logarithm with base *e*
*P* ( *A* ) power set (set of all subsets) of *A*
**1** *{* *predicate* *}* indicator function ( **1** *{* *predicate* *}* = . 1 if the *predicate* is true, else 0)
*←* assignment

analysis

***∇*** *f* ( ***x*** ) *∈* **R** *[n]* gradient of a function *f* : **R** *[n]* *→* **R** at a point ***x*** *∈* **R** *[n]*

x

***Dg*** ( ***x*** ) *∈* **R** *[m]* *[×]* *[n]* Jacobian of a function ***g*** : **R** *[n]* *→* **R** *[m]* at a point ***x*** *∈* **R** *[n]*

***H*** *f* ( ***x*** ) *∈* **R** *[n]* *[×]* *[n]* Hessian of a function *f* : **R** *[n]* *→* **R** at a point ***x*** *∈* **R** *[n]*

*f* ( *n* )
*f* *∈O* ( *g* ) *f* grows at most as fast as *g* (up to constant factors), 0 *≤* lim sup *n* *→* ∞ ��� *g* ( *n* )

*f* *∈* *O* [˜] ( *g* ) *f* grows at most as fast as *g* up to constant and logarithmic factors

*∥·∥* *α* *α* -norm
*∥·∥* ***A*** Mahalanobis norm induced by matrix ***A***

linear algebra

***I*** identity matrix

***A*** *[⊤]* transpose of matrix ***A***
***A*** *[−]* [1] inverse of invertible matrix ***A***
***A*** [1] [/] [2] square root of a symmetric and positive semi-definite matrix ***A***

det ( ***A*** ) determinant of ***A***
tr ( ***A*** ) trace of ***A***, ∑ *i* ***A*** ( *i*, *i* )
diag *i* *∈* *I* *{* *a* *i* *}* diagonal matrix with elements *a* *i*, indexed according to the set *I*

probability

Ω sample space
*A* event space
**P** probability measure

*X* *∼* *P* random variable *X* follows the distribution *P*


*<* ∞
���


iid
*X* 1: *n* *∼* *P* random variables *X* 1: *n* are independent and identically distributed according to

distribution *P*

*x* *∼* *P* value *x* is sampled according to distribution *P*
*P* *X* cumulative distribution function of a random variable *X*

*P* *X* tail distribution function of a random variable *X*
*P* *X* *[−]* [1] quantile function of a random variable *X*
*p* *X* probability mass function (if discrete) or probability density function
(if continuous) of a random variable *X*
∆ *[A]* set of all probability distributions over the set *A*
*δ* *α* Dirac delta function, point density at *α*

*X* *⊥* *Y* random variable *X* is independent of random variable *Y*
*X* *⊥* *Y* *|* *Z* random variable *X* is conditionally independent of random variable *Y*
given random variable *Z*

**E** [ *X* ] expected value of random variable *X*
**E** *x* *∼* *X* [ *f* ( *x* )] expected value of the random variable *f* ( *X* ), **E** [ *f* ( *X* )]
**E** [ *X* *|* *Y* ] conditional expectation of random variable *X* given random variable *Y*
Cov [ *X*, *Y* ] covariance of random variable *X* and random variable *Y*

xi


Cor [ *X*, *Y* ] correlation of random variable *X* and random variable *Y*
Var [ *X* ] variance of random variable *X*
Var [ *X* *|* *Y* ] conditional variance of random variable *X* given random variable *Y*
***Σ*** **X** covariance matrix of random vector **X**
***Λ*** **X** precision matrix of random vector **X**

MSE ( *X* ) mean squared error of random variable *X*
*X* *n* sample mean of random variable *X* with *n* samples
*S* *n* [2] sample variance of random variable *X* with *n* samples

*X* *n* a.s. *→* *X* the sequence of random variables *X* *n* converges almost surely to *X*

**P**
*X* *n* *→* *X* the sequence of random variables *X* *n* converges to *X* in probability

*D*
*X* *n* *→* *X* the sequence of random variables *X* *n* converges to *X* in distribution

S [ *u* ] surprise associated with an event of probability *u*
H [ *p* ], H [ *X* ] entropy of distribution *p* (or random variable *X* )
H [ *p* *∥* *q* ] cross-entropy of distribution *q* with respect to distribution *p*
KL ( *p* *∥* *q* ) KL-divergence of distribution *q* with respect to distribution *p*
H [ *X* *|* *Y* ] conditional entropy of random variable *X* given random variable *Y*
H [ *X*, *Y* ] joint entropy of random variables *X* and *Y*
I ( *X* ; *Y* ) mutual information of random variables *X* and *Y*
I ( *X* ; *Y* *|* *Z* ) conditional mutual information of random variables *X* and *Y* given random

variable *Z*

*N* ( ***µ***, ***Σ*** ) normal distribution with parameters ***µ*** and ***Σ***
Laplace ( ***µ***, *h* ) Laplace distribution with parameters ***µ*** and *h*
Unif ( *S* ) uniform distribution on the set *S*
Bern ( *p* ) Bernoulli distribution with parameter *p*
Bin ( *n*, *p* ) binomial distribution with parameters *n* and *p*
Beta ( *α*, *β* ) beta distribution with parameters *α* and *β*

supervised learning

***θ*** parameterization of a model

*X* input space
*Y* label space
***x*** *∈X* input
*ϵ* ( ***x*** ) zero-mean noise, sometimes assumed to be independent of ***x***
*y* *∈Y* (noisy) label, *f* ( ***x*** ) + *ϵ* ( ***x*** ) where *f* is unknown
*D ⊆X × Y* labeled training data, *{* ( ***x*** *i*, *y* *i* ) *}* *i* *[n]* = 1
***X*** *∈* **R** *[n]* *[×]* *[d]* design matrix when *X* = **R** *[d]*

***Φ*** *∈* **R** *[n]* *[×]* *[e]* design matrix in feature space **R** *[e]*

***y*** *∈* **R** *[n]* label vector when *Y* = **R**

*p* ( ***θ*** ) prior belief about ***θ***
*p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) posterior belief about ***θ*** given training data

xii

*p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) likelihood of training data under the model parameterized by ***θ***
*p* ( *y* 1: *n* *|* ***x*** 1: *n* ) marginal likelihood of training data

***θ*** ˆ MLE maximum likelihood estimate of ***θ***
***θ*** ˆ MAP maximum a posteriori estimate of ***θ***

*ℓ* nll ( ***θ*** ; *D* ) negative log-likelihood of the training data *D* under model ***θ***

bayesian linear models

***w*** *∈* **R** *[d]* weights of linear function *f* ( ***x*** ; ***w*** ) = ***w*** *[⊤]* ***x***

***w*** ˆ ls least squares estimate of ***w***
***w*** ˆ ridge ridge estimate of ***w***
***w*** ˆ lasso lasso estimate of ***w***

*N* ( **0**, *σ* *p* [2] ***[I]*** [)] prior
*N* ( ***w*** *[⊤]* ***x***, *σ* *n* [2] [)] likelihood
***µ*** *∈* **R** *[d]* posterior mean, *σ* *n* *[−]* [2] ***[Σ][X]*** *[⊤]* ***[y]***
***Σ*** *∈* **R** *[d]* *[×]* *[d]* posterior covariance matrix, ( *σ* *n* *[−]* [2] ***[X]*** *[⊤]* ***[X]*** [ +] *[ σ]* *p* *[−]* [2] ***[I]*** [)] *[−]* [1]

***K*** *∈* **R** *[n]* *[×]* *[n]* kernel matrix, *σ* [2]
*p* ***[XX]*** *[⊤]*

*σ* logistic function
Bern ( *σ* ( ***w*** *[⊤]* ***x*** )) logistic likelihood
*ℓ* log ( *·* ; ***x***, *y* ) logistic loss of a single training example ( ***x***, *y* )

kalman filters

**X** *t* sequence of hidden states in **R** *[d]*

**Y** *t* sequence of observations in **R** *[m]*

***F*** *∈* **R** *[d]* *[×]* *[d]* motion model
***H*** *∈* **R** *[m]* *[×]* *[d]* sensor model

***ϵ*** *t* zero-mean motion noise with covariance matrix ***Σ*** *x*
***η*** *t* zero-mean sensor noise with covariance matrix ***Σ*** *y*

***K*** *t* *∈* **R** *[d]* *[×]* *[m]* Kalman gain

gaussian processes

*µ* : *X →* **R** mean function
*k* : *X × X →* **R** kernel function / covariance function
*f* *∼GP* ( *µ*, *k* ) *f* is a Gaussian process with mean function *µ* and kernel function *k*

*H* *k* ( *X* ) reproducing kernel Hilbert space associated with kernel function *k* : *X × X →* **R**

xiii


deep models

***W*** *l* *∈* **R** *[n]* *[l]* *[×]* *[n]* *[l]* *[−]* [1] weight matrix of layer *l*
***ν*** [(] *[l]* [)] *∈* **R** *[n]* *[l]* activations of layer *l*

*φ* activation function
Tanh hyperbolic tangent activation function
ReLU rectified linear unit activation function

*σ* *i* ( ***f*** ) softmax function computing the probability mass of class *i* given outputs ***f***

variational inference

*Q* variational family
***λ*** *∈* Λ variational parameters
*q* ***λ*** variational posterior parameterized by ***λ***
*L* ( *q*, *p* ; *D* ) evidence lower bound for data *D* of variational posterior *q* and true posterior *p* ( *· | D* )

markov chains

*S* set of *n* states

*X* *t* sequence of states
*p* ( *x* *[′]* *|* *x* ) transition function, probability of going from state *x* to state *x* *[′]*

*p* [(] *[t]* [)] ( *x* *[′]* *|* *x* ) probability of reaching *x* *[′]* from *x* in exactly *t* steps
***P*** *∈* **R** *[n]* *[×]* *[n]* transition matrix

*q* *t* distribution over states at time *t*
*π* stationary distribution

*∥* *µ* *−* *ν* *∥* TV total variation distance between two distributions *µ* and *ν*
*τ* TV mixing time with respect to total variation distance

markov chain monte carlo methods

*r* ( ***x*** *[′]* *|* ***x*** ) proposal distribution, probability of proposing ***x*** *[′]* when in ***x***
*α* ( ***x*** *[′]* *|* ***x*** ) acceptance distribution, probability of accepting the proposal ***x*** *[′]* when in ***x***

*f* energy function

active learning

*S* *⊆X* set of observations
*I* ( *S* ) maximization objective, quantifying the “information value” of *S*
∆ *I* ( ***x*** *|* *A* ) marginal gain of observation ***x*** with respect to objective *I* given prior observations *A*

xiv

bayesian optimization

*R* *T* cumulative regret for time horizon *T*
*F* ( ***x*** ; *µ*, *σ* ) acquisition function
*γ* *T* maximum information gain after *T* rounds

UCB *t* ( ***x*** ) upper confidence bound acquisition function
PI *t* ( ***x*** ) probability of improvement acquisition function
EI *t* ( ***x*** ) expected improvement acquisition function
*π* ( ***x*** *| D* ) posterior probability of ***x*** maximizing *f*, used in probability matching

reinforcement learning

*X*, *X* set of states

*A*, *A* set of actions
*p* ( *x* *[′]* *|* *x*, *a* ) dynamics model, probability of transitioning from state *x* to state *x* *[′]* when playing

action *a*

*r* reward function

*X* *t* sequence of states
*A* *t* sequence of actions
*R* *t* sequence of rewards
*π* ( *a* *|* *x* ) policy, probability of playing action *a* when in state *x*
*G* *t* discounted payoff from time *t*
*γ* discount factor

*v* *[π]* *t* [(] *[x]* [)] state value function, average discounted payoff from time *t* starting from state *x*
*q* *[π]* *t* [(] *[x]* [,] *[ a]* [)] state-action value function, average discounted payoff from time *t* starting from state *x*
playing action *a*
*a* *[π]* *t* [(] *[x]* [,] *[ a]* [)] advantage function, *q* *[π]* *t* [(] *[x]* [,] *[ a]* [)] *[ −]* *[v]* *[π]* *t* [(] *[x]* [)]
*j* ( *π* ) policy value function, expected reward of policy *π*

### *1* *Fundamentals*

*1.1*
*Probability*

Probabilities are commonly interpreted in two different ways. In the
frequentist interpretation, one interprets the probability of an event
(say a coin coming up “heads” when flipping it) as the limit of relative
frequencies in repeated independent experiments. That is,

# events ha pp enin g in *N* trials
Probability = lim .
*N* *→* ∞ *N*

This interpretation is natural, but has a few issues. Often, probabilities
are used to describe *beliefs* that one has about the outcome of a trial.
While it is natural to consider the relative frequency of the outcome in
repeated experiments as our belief, it is not very difficult to conceive of
settings where repeated experiments do not make sense. Consider the
outcome: “person X will live for at least 80 years.” There is no way we
can conduct multiple independent experiments. In these settings, our
notion of probability is simply a (subjective) measure of uncertainty
about occurrence of events and does not require any experiments. This
notion is commonly understood as *Bayesian reasoning* or the Bayesian
interpretation of probability. [1] 1 In the early 20th century, Bruno De
Finetti has done foundational work to

formalize this notion.


2 probabilistic artificial intelligence

*1.1.1* *Probability Spaces*

A probability space is a mathematical model for a random experiment.
The set of all possible outcomes of the experiment Ω is called *sample*
*space* . An *event A* *⊆* Ω of interest may be any combination of possible
outcomes. The set of all events *A ⊆P* ( Ω ) that we are interested in
is often called the *event space* of the experiment. [2] This set of events is 2 We use *P* ( Ω ) to denote the *power set*
required to be a *σ* -algebra over the sample space. (set of all subsets) of Ω.

**Definition 1.1** ( *σ* -algebra) **.** Given the set Ω, the set *A ⊆P* ( Ω ) is a
*σ-algebra* over Ω if the following properties are satisfied:

1. Ω *∈A* ;
2. if *A* *∈A*, then *A* *∈A* ( *closedness under complements* ); and
3. if we have *A* *i* *∈A* for all *i*, then [�] *i* [∞] = 1 *[A]* *[i]* *[∈A]* [ (] *[closedness under]*
*countable unions* ).

Note that the three properties of *σ* -algebras correspond to characteristics we universally expect when working with random experiments.
Namely, that we are able to reason about the event Ω that any of the
possible outcomes occur, that we are able to reason about an event
not occurring, and that we are able to reason about events that are
composed of multiple (smaller) events.



**Definition 1.3** (Probability measure) **.** Given the set Ω and the *σ* -algebra
*A* over Ω, the function

**P** : *A →* **R**

is a *probability measure* on *A* if the *Kolmogorov axioms* are satisfied:

1. 0 *≤* **P** ( *A* ) *≤* 1 for any *A* *∈A* ;
2. **P** ( Ω ) = 1; and
3. **P** ( [�] *i* [∞] = 1 *[A]* *[i]* [) =] [ ∑] *i* [∞] = 1 **[P]** [(] *[A]* *[i]* [)] [ for any countable set of mutually disjoint]
events *{* *A* *i* *∈A}* *i* . [3] 3 We say that a set of sets *{* *A* *i* *}* *i* is disjoint
if for all *i* *̸* = *j* we have *A* *i* *∩* *A* *j* = ∅.
Remarkably, all further statements about probability follow from these

fundamentals 3


three natural axioms. For an event *A* *∈A*, we call **P** ( *A* ) the *probability*
of *A* . We are now ready to define a probability space.

**Definition 1.4** (Probability space) **.** A *probability space* is a triple ( Ω, *A*, **P** )

where

- Ω is a sample space,

- *A* is a *σ* -algebra over Ω, and

- **P** is a probability measure on *A* .









*1.1.2* *Random Variables*

The set Ω is often rather complex. For example, take Ω to be the set of
all possible graphs on *n* vertices. Then the outcome of our experiment
is a graph. Usually, we are not interested in a specific graph but rather
a property such as the number of edges, which is shared by many
graphs. A function that maps a graph to its number of edges is a

4 probabilistic artificial intelligence

random variable.

**Definition 1.7** (Random variable) **.** A *random variable X* is a function

*X* : Ω *→T*


where *T* is called *target space* of the random variable, [4] and where *X* 4 For a random variable that maps a

respects the information available in the *σ* -algebra *A* . That is, [5] graph to its number of edges, *T* = **N** 0 .

For our purposes, you can generally assume *T ⊆* **R** .


respects the information available in the *σ* -algebra *A* . That is, [5]


*∀* *S* *⊆T* : *{* *ω* *∈* Ω : *X* ( *ω* ) *∈* *S* *} ∈A* . (1.2)


5 In our example of throwing a die, *X*
should assign the same value to the outcomes 1, 3, 5.


Concrete values *x* of a random variable *X* are often referred to as *states*

or *realizations* of *X* . The probability that *X* takes on a value in *S* *⊆T* is

**P** ( *X* *∈* *S* ) = **P** ( *{* *ω* *∈* Ω : *X* ( *ω* ) *∈* *S* *}* ) . (1.3)

*1.1.3* *Distributions*

Consider a random variable *X* on a probability space ( Ω, *A*, **P** ), where
Ω is a compact subset of **R**, and *A* the Borel *σ* -algebra.

In this case, we can refer to the probability that *X* assumes a particular
state or set of states by writing

*p* *X* ( *x* ) = . **P** ( *X* = *x* ) (in the discrete setting), (1.4)

*P* *X* ( *x* ) = . **P** ( *X* *≤* *x* ) . (1.5)

Note that “ *X* = *x* ” and “ *X* *≤* *x* ” are merely events (that is, they characterize subsets of the sample space Ω satisfying this condition) which
are in the Borel *σ* -algebra, and hence their probability is well-defined.

Hereby, *p* *X* and *P* *X* are referred to as the probability mass function
(PMF) and cumulative distribution function (CDF) of *X*, respectively.
Note that we can also *implicitly* define probability spaces through random variables and their associated PMF/CDF, which is often very con
venient.

Further, note that for continuous variables, **P** ( *X* = *x* ) = 0. Here, instead we typically use the probability density function (PDF), to which
we (with slight abuse of notation) also refer with *p* *X* . We discuss densities in greater detail in section 1.1.4.


fundamentals 5





We call the subset *S* *⊆T* of the domain of a PMF or PDF *p* *X* such that
all elements *x* *∈* *S* have positive probability, *p* *X* ( *x* ) *>* 0, the *support* of
the distribution *p* *X* . This quantity is denoted by *X* ( Ω ) .

6 probabilistic artificial intelligence

*1.1.4* *Continuous Distributions*

As mentioned, a continuous random variable can be characterized by
its *probability density function* (PDF). But what is a density? We can
derive some intuition from physics.

Let *M* be a (non-homogeneous) physical object, e.g., a rock. We commonly use *m* ( *M* ) and vol ( *M* ) to refer to its mass and volume, respectively. Now, consider for a point ***x*** *∈* *M* and a ball *B* *r* ( ***x*** ) around ***x*** with
radius *r* the following quantities:

*r* lim *→* 0 [vol] [(] *[B]* *[r]* [(] ***[x]*** [)) =] [ 0] *r* lim *→* 0 *[m]* [(] *[B]* *[r]* [(] ***[x]*** [)) =] [ 0.]

They appear utterly uninteresting at first, yet, if we divide them, we
get what is called the *density* of *M* at ***x*** .


*m* ( *B* *r* ( ***x*** ))
lim
*r* *→* 0 vol ( *B* *r* ( ***x*** ))


= . *ρ* ( ***x*** ) .


We know that the relationship between density and mass is described
by the following formula:


*m* ( *M* ) =
�


*M* *[ρ]* [(] ***[x]*** [)] *[ d]* ***[x]*** [.]


In other words, the density is to be integrated. For a small region *I*
around ***x***, we can approximate *m* ( *I* ) *≈* *ρ* ( ***x*** ) *·* vol ( *I* ) .

Crucially, observe that even though the mass of any particular point
***x*** is zero, i.e., *m* ( *{* ***x*** *}* ) = 0, assigning a density *ρ* ( ***x*** ) to ***x*** is useful for
integration and approximation. The same idea applies to continuous
random variables, only that volume corresponds to intervals on the
real line and mass to probability. Recall that probability density functions are normalized such that their probability mass across the entire
real line integrates to one.





*N* ( *x* ; 0, 1 )

0.4

0.3

0.2

0.1

0.0

*−* 2 0 2

*x*

Figure 1.1: PDF of the standard normal
distribution. Observe that the PDF is
symmetric around the mode.




fundamentals 7

**Uniform distribution** The (continuous) *uniform distribution* Unif ([ *a*, *b* ]) is
the only distribution that assigns constant density to all points in the support

[ *a*, *b* ] . That is, it has PDF




*p* ( *u* ) =

and CDF

*P* ( *u* ) =


*b* *−* 1 *a* if *u* *∈* [ *a*, *b* ]

0 otherwise
�

*u* *b* *−−* *a* *a* if *u* *∈* [ *a*, *b* ]

0 otherwise.
�



8 probabilistic artificial intelligence



*1.1.5* *Joint Probability*

A joint probability (as opposed to a marginal probability) is the probability of two or more events occurring simultaneously:

**P** ( *A*, *B* ) = . **P** ( *A* *∩* *B* ) . (1.9)

In terms of random variables, this concept extends to joint distributions. Instead of characterizing a single random variable, a *joint dis-*
*tribution* is a function *p* **X** : **R** *[n]* *→* **R**, characterizing a *random vector*
**X** = [ . *X* 1 *· · ·* *X* *n* ] *⊤* . For example, if the *X* *i* are discrete, the joint distribution characterizes joint probabilities of the form

**P** ( **X** = [ *x* 1, . . ., *x* *n* ]) = **P** ( *X* 1 = *x* 1, . . ., *X* *n* = *x* *n* ),

and hence describes the relationship among all variables *X* *i* . We use
*X* *i* : *j* to denote the random vector [ *X* *i* *· · ·* *X* *j* ] *[⊤]* .

We can “sum out” (respectively “integrate out”) variables from a joint
distribution in a process called “marginalization”.





*1.1.6* *Conditional Probability*

Conditional probability updates the probability of an event *A* given
some new information, for example, after observing the event *B* .

**Definition 1.17** (Conditional probability) **.** Given two events *A* and *B*
such that **P** ( *B* ) *>* 0, the probability of *A* conditioned on *B* is given as

**P** ( *A* *|* *B* ) = . **P** ( *A*, *B* ) (1.11)

**P** ( *B* ) [.]

Simply rearranging the terms yields,

**P** ( *A*, *B* ) = **P** ( *A* *|* *B* ) *·* **P** ( *B* ) = **P** ( *B* *|* *A* ) *·* **P** ( *A* ) . (1.12)


Figure 1.2: Conditioning an event *A* on
another event *B* can be understood as replacing the universe of all possible outcomes Ω by the observed outcomes *B* .
Then, the conditional probability is simply expressing the likelihood of *A* given
that *B* occurred.




Thus, the probability that both *A* and *B* occur can be calculated by
multiplying the probability of event *A* and the probability of *B* conditional on *A* occurring.

We say **Z** *∼* **X** *|* **Y** = ***y*** (or simply **Z** *∼* **X** *|* ***y*** ) if **Z** follows the *conditional*

*distribution*

*p* **X** *|* **Y** ( ***x*** *|* ***y*** ) = . *[p]* **X**, **Y** [(] ***[x]*** [,] ***[y]*** [)] . (1.13)

*p* **Y** ( ***y*** )

If *X* and *Y* are discrete, we have that *p* *X* *|* *Y* ( *x* *|* *y* ) = **P** ( *X* = *x* *|* *Y* = *y* )
as one would naturally expect.



Extending eq. (1.12) to arbitrary random vectors yields the product
rule (also called *chain rule of probability* ).


fundamentals 9

As noted previously, if the correspondence between the random vectors **X**

and **Y** and their states ***x*** and ***y*** is clear
from context, we often omit the indices
of the PDF *p* and simply write *p* ( ***x*** *|* ***y*** ),
*p* ( ***x***, ***y*** ) and *p* ( ***y*** ) . However, this does *not*
mean that they refer to the same distribution *p* .



Combining sum rule and product rule, we can compute marginal
probabilities too. If **Y** is continuous, then


*p* ( ***x*** ) =
�


**Y** ( Ω ) *[p]* [(] ***[x]*** [,] ***[ y]*** [)] *[ d]* ***[y]*** [ =] �


**Y** ( Ω ) *[p]* [(] ***[x]*** *[ |]* ***[ y]*** [)] *[ ·]* *[ p]* [(] ***[y]*** [)] *[ d]* ***[y]*** (1.15) first using the sum rule (product rule (1.14) 1.10) then the


Analogously, one can condition on a discrete random variable by replacing the integral with a sum. The above is called the *law of total*
*probability* (LOTP). Colloquially, this is often referred to as to “condition” on **Y** . If it is difficult to compute *p* ( ***x*** ) directly, conditioning can
be a useful technique when **Y** is chosen such that the densities *p* ( ***x*** *|* ***y*** )
and *p* ( ***y*** ) are straightforward to understand.


10 probabilistic artificial intelligence




*1.1.7* *Independence*

Two random vectors **X** and **Y** are *independent* (denoted **X** *⊥* **Y** ) iff
knowledge about the state of one random vector does not affect the
distribution of the other random vector, namely if their conditional
CDF (or in case they have a joint density, their conditional PDF) simplifies to

*P* **X** *|* **Y** ( ***x*** *|* ***y*** ) = *P* **X** ( ***x*** ), *p* **X** *|* **Y** ( ***x*** *|* ***y*** ) = *p* **X** ( ***x*** ) . (1.17)

For the conditional probabilities to be well-defined, we need to assume
that **P** ( **Y** = ***y*** ) *>* 0.

The more general characterization of independence is that **X** and **Y** are
independent if and only if their joint CDF (or in case they have a joint
density, their joint PDF) can be decomposed as follows:

*P* **X**, **Y** ( ***x***, ***y*** ) = *P* **X** ( ***x*** ) *·* *P* **Y** ( ***y*** ), *p* **X**, **Y** ( ***x***, ***y*** ) = *p* **X** ( ***x*** ) *·* *p* **Y** ( ***y*** ) . (1.18)

The equivalence of the two characterizations (when **P** ( ***y*** ) *>* 0) is easily
proven using the product rule: *P* **X**, **Y** ( ***x***, ***y*** ) = *P* **Y** ( ***y*** ) *·* *P* **X** *|* **Y** ( ***x*** *|* ***y*** ) .

We often say that a set of random variables *{* *X* *i* *}* *i* is i.i.d. to abbreviate
that the *X* *i* are “(mutually) independent and identically distributed”,

iid
and we write *X* *i* *∼* *p* .

A “weaker” notion of independence is conditional independence. [7] 7 It is explained in remark 1.22 how
“weaker” is to be interpreted in this con
text.

fundamentals 11

Two random vectors **X** and **Y** are *conditionally independent* given a random vector **Z** (denoted **X** *⊥* **Y** *|* **Z** ) iff, given **Z**, knowledge about the

value of one random vector **Y** does not affect the distribution of the

other random vector **X**, namely if

*P* **X** *|* **Y**, **Z** ( ***x*** *|* ***y***, ***z*** ) = *P* **X** *|* **Z** ( ***x*** *|* ***z*** ), (1.19a)

*p* **X** *|* **Y**, **Z** ( ***x*** *|* ***y***, ***z*** ) = *p* **X** *|* **Z** ( ***x*** *|* ***z*** ) . (1.19b)

Similarly to independence, we have that **X** and **Y** are conditionally
independent given **Z** if and only if their joint CDF or joint PDF can be
decomposed as follows:

*P* **X**, **Y** *|* **Z** ( ***x***, ***y*** *|* ***z*** ) = *P* **X** *|* **Z** ( ***x*** *|* ***z*** ) *·* *P* **Y** *|* **Z** ( ***y*** *|* ***z*** ), (1.20a)

*p* **X**, **Y** *|* **Z** ( ***x***, ***y*** *|* ***z*** ) = *p* **X** *|* **Z** ( ***x*** *|* ***z*** ) *·* *p* **Y** *|* **Z** ( ***y*** *|* ***z*** ) . (1.20b)

the value of *Y*, and hence, *X* *̸⊥* *Y* *|* *X* +

The independence and conditional independence of events is defined
analogously.

*1.1.8* *Directed Graphical Models*


Directed graphical models (also called *Bayesian networks* ) are often
used to visually denote the (conditional) independence relationships
of a large number of random variables. They are a schematic representation (as a directed acyclic graph) of the factorization of the joint
distribution into a product of conditional distributions. Given the sequence of random variables *{* *X* *i* *}* *i* *[n]* = 1 [, their joint distribution can be]
expressed as

*n*
###### P ( X 1 = x 1, . . ., X n = x n ) = ∏ P ( X i = x i | parents ( X i )) (1.21)

*i* = 1


|c Y X1 · · · Xn a1 an|c|
|---|---|
|a1||


Figure 1.3: Example of a directed
graphical model. The random variables *X* 1, . . ., *X* *n* are mutually independent given the random variable *Y* . The
squared rectangular nodes are used to
represent dependencies on parameters

*c*, *a* 1, . . ., *a* *n* .

12 probabilistic artificial intelligence

where parents ( *X* *i* ) is the set of parents of the vertex *X* *i* in the directed
graphical model. In other words, the parenthood relationship encodes
a conditional independence of a random variable *X* with a random
variable *Y* given their parents, [10] 10 More generally, vertices *u* and *v* are
conditionally independent given a set of
*X* *⊥* *Y* *|* parents ( *X* ), parents ( *Y* ) . (1.22) vertices *Z* if *Z d-separates u* and *v*, which
we will not cover in depth here.

Thus, eq. (1.21) simply uses the product rule and the conditional independence relationships to factorize the joint distribution.


An example of a directed graphical model is given in fig. 1.3. Circular vertices represent random quantities (i.e., random variables). In
contrast, square vertices are commonly used to represent deterministic quantities (i.e., parameters that the distributions depend on). In
the given example, we have that *X* *i* is conditionally independent of all
other *X* *j* given *Y* .

*Plate notation* is a condensed notation used to represent repeated variables of a graphical model. An example is given in fig. 1.4.

*1.1.9* *Expectation*

The *expected value* **E** [ **X** ] of a random vector **X** is the (asymptotic) arithmetic mean of an arbitrarily increasing number of independent real
izations of **X** . We have
###### E [ X ] = . ∑ x · p ( x ) (1.23a)

***x*** *∈* **X** ( Ω )






Figure 1.4: The same directed graphical
model as in fig. 1.3 using plate notation.


**E** [ **X** ] = .
�


**X** ( Ω ) ***[x]*** *[ ·]* *[ p]* [(] ***[x]*** [)] *[ d]* ***[x]*** (1.23b)


if **X** is discrete or continuous, respectively. [11] A very special and often 11 In infinite probability spaces, absolute
used property of expectations is their *linearity* . For any random vectors convergence of **E** [ **X** ] is necessary for the
existence of **E** [ **X** ] .
**X** and **Y** in **R** *[n]* and any ***A*** *∈* **R** *[m]* *[×]* *[n]*, ***b*** *∈* **R** *[m]* we have

**E** [ ***A*** **X** + ***b*** ] = ***A*** **E** [ **X** ] + ***b*** and (1.24)

**E** [ **X** + **Y** ] = **E** [ **X** ] + **E** [ **Y** ] . (1.25)

Note that **X** and **Y** do not necessarily have to be independent! Further,
if **X** and **Y** are independent then

**E** � **XY** *[⊤]* [�] = **E** [ **X** ] *·* **E** [ **Y** ] *[⊤]* . (1.26)


fundamentals 13







and *p* ( ***x*** ) does not depend on *t* .






**Gradient** The *gradient* of a function *f* :
**R** *[n]* *→* **R** at a point ***x*** *∈* **R** *[n]* is


***∇*** *f* ( ***x*** ) = . � *∂∂* ***x*** *f* ( ( 1 ***x*** ) )


*∂∂* ***x*** *f* ( ( 1 ***x*** ) ) *· · ·* *∂∂* ***x*** *f* ( ( *n* ***x*** ) )


*⊤*

.
�


The following intuitive lemma can be used to compute expectations of ***x*** ***x*** *n* (1.29)
transformed random variables.




isfied in most cases) is sufficient.



This is a nontrivial fact that can be proven using the change of variables

formula discussed in section 1.1.11.

Similarly to conditional probability, we can also define conditional expectations. The expectation of a continuous random vector **X** given
that **Y** = ***y*** is defined as


**E** [ **X** *|* **Y** = ***y*** ] = . (1.31)
� **X** ( Ω ) ***[x]*** *[ ·]* *[ p]* **[X]** *[|]* **[Y]** [(] ***[x]*** *[ |]* ***[ y]*** [)] *[ d]* ***[x]*** [.]

14 probabilistic artificial intelligence

The definition is analogous for discrete random variables. Note that
**E** [ **X** *|* **Y** = *·* ] defines a deterministic mapping from ***y*** to **E** [ **X** *|* **Y** = ***y*** ] .
Therefore, **E** [ **X** *|* **Y** ] is itself a random vector:

**E** [ **X** *|* **Y** ]( *ω* ) = **E** [ **X** *|* **Y** = **Y** ( *ω* )] (1.32)

where *ω* *∈* Ω. This random vector **E** [ **X** *|* **Y** ] is called the *conditional*
*expectation* of **X** given **Y** .



Here, the expression **E** [ **E** [ **X** *|* **Y** ]] first averages over **Y**, then over **X** .

*Proof.* We only prove the case where **X** and **Y** have a joint density. We

have

**E** [ **X** *|* **Y** = ***y*** ] = ***x*** *·* *p* ( ***x*** *|* ***y*** ) *d* ***x***
�

and hence

**E** [ **E** [ **X** *|* **Y** ]] = ***x*** *·* *p* ( ***x*** *|* ***y*** ) *d* ***x*** *p* ( ***y*** ) *d* ***y***
� [�] � �

= ***x*** *·* *p* ( ***x***, ***y*** ) *d* ***x*** *d* ***y*** by definition of conditional densities
�� (1.13)
= ***x*** *p* ( ***x***, ***y*** ) *d* ***y*** *d* ***x*** by Fubini’s theorem
� �

= ***x*** *·* *p* ( ***x*** ) *d* ***x*** using the sum rule (1.10)
�

= **E** [ **X** ] .

Note that the tower rule can be used analogously to the law of total
probability (1.15) to “condition” on a random vector **Y** . For this reason,
the tower rule is also known as the *law of total expectation* (LOTE).


fundamentals 15


*1.1.10* *Covariance and Variance*

Covariance measures the linear dependence between two random vectors. Given two random vectors **X** in **R** *[n]* and **Y** in **R** *[m]*, their *covariance*

is given as

Cov [ **X**, **Y** ] = . **E** � ( **X** *−* **E** [ **X** ])( **Y** *−* **E** [ **Y** ]) *[⊤]* [�] (1.35)

= **E** � **XY** *[⊤]* [�] *−* **E** [ **X** ] *·* **E** [ **Y** ] *[⊤]* (1.36)

= Cov [ **Y**, **X** ] *[⊤]* *∈* **R** *[n]* *[×]* *[m]* . (1.37)

A direct consequence of the definition of covariance (1.35) is that given
linear maps ***A*** *∈* **R** *[n]* *[′]* *[×]* *[n]*, ***B*** *∈* **R** *[m]* *[′]* *[×]* *[m]*, vectors ***c*** *∈* **R** *[n]* *[′]*, ***d*** *∈* **R** *[m]* *[′]* and
random vectors **X** in **R** *[n]* and **Y** in **R** *[m]*, we have that

Cov [ ***A*** **X** + ***c***, ***B*** **Y** + ***d*** ] = ***A*** Cov [ **X**, **Y** ] ***B*** *[⊤]* . (1.38)

Two random vectors **X** and **Y** are said to be *uncorrelated* iff Cov [ **X**, **Y** ] =
**0** . Note that if **X** and **Y** are independent, then eq. (1.26) implies that **X**
and **Y** are uncorrelated. The reverse does not hold in general.









- Cov [ *X*, *Y* ] is symmetric,

- Cov [ *X*, *Y* ] is linear (here we use
**E** *X* = **E** *Y* = 0), and

- Cov [ *X*, *X* ] *≥* 0.







(1.41) using the Euclidean inner product

16 probabilistic artificial intelligence



The variance is a measure of uncertainty about the value of a random
vector. Given the random vector **X** in **R** *[n]*, its *variance* is given as

Var [ **X** ] = . Cov [ **X**, **X** ] (1.43)

= **E** � ( **X** *−* **E** [ **X** ])( **X** *−* **E** [ **X** ]) *[⊤]* [�] (1.44)

= **E** � **XX** *[⊤]* [�] *−* **E** [ **X** ] *·* **E** [ **X** ] *[⊤]* (1.45)




 . (1.46)


=







Cov [ *X* 1, *X* 1 ] *· · ·* Cov [ *X* 1, *X* *n* ]

... ... ...

Cov [ *X* *n*, *X* 1 ] *· · ·* Cov [ *X* *n*, *X* *n* ]


The variance of a random vector **X** is also called the *covariance matrix*

of **X** and denoted by ***Σ*** **X** (or ***Σ*** if the correspondence to **X** is clear from
context). A covariance matrix is symmetric by definition due to the
symmetry of covariance.



Two useful properties of variance are the following:

- It follows from eq. (1.38) that for any linear map ***A*** *∈* **R** *[m]* *[×]* *[n]* and
vector ***b*** *∈* **R** *[m]*,

Var [ ***A*** **X** + ***b*** ] = ***A*** Var [ **X** ] ***A*** *[⊤]* . (1.47)

In particular, Var [ *−* **X** ] = Var [ **X** ] .

- It follows from the definition of variance (1.44) that for any two
random vectors **X** and **Y**,

Var [ **X** + **Y** ] = Var [ **X** ] + Var [ **Y** ] + 2Cov [ **X**, **Y** ] . (1.48)

In particular, if **X** and **Y** are independent then the covariance term
vanishes and Var [ **X** + **Y** ] = Var [ **X** ] + Var [ **Y** ] .

fundamentals 17


Analogously to conditional probability and conditional expectation,
we can also define conditional variance. The *conditional variance* of a
random vector **X** given another random vector **Y** is

Var [ **X** *|* **Y** ] = . **E** � ( **X** *−* **E** [ **X** *|* **Y** ])( **X** *−* **E** [ **X** *|* **Y** ]) *[⊤]* [��] � **Y** � . (1.49)

Intuitively, the conditional variance is the remaining variance when we
use **E** [ **X** *|* **Y** ] to predict **X** rather than if we used **E** [ **X** ] .



Here, the first term measures the average deviation from the mean of **X**
across realizations of **Y** and the second term measures the uncertainty
in the mean of **X** across realizations of **Y** . In section 2.2, we will see

that both terms have a meaningful characterization in the context of
Bayesian learning.

*Proof.* To simplify the notation, we present only a proof for the univariate setting. [15] 15 In the *univariate* setting (as opposed
to the *multivariate* setting) we consider a
Var [ *X* ] = **E** *X* [2] [�] *−* **E** [ *X* ] [2] single random variable.
�

= **E** � **E** � *X* [2] *|* *Y* �� *−* **E** [ **E** [ *X* *|* *Y* ]] [2] by the tower rule (1.33)

= **E** �Var [ *X* *|* *Y* ] + **E** [ *X* *|* *Y* ] [2] [�] *−* **E** [ **E** [ *X* *|* *Y* ]] [2] by the definition of variance (1.45)

= **E** [ Var [ *X* *|* *Y* ]] + � **E** � **E** [ *X* *|* *Y* ] [2] [�] *−* **E** [ **E** [ *X* *|* *Y* ]] [2] [�]

= **E** [ Var [ *X* *|* *Y* ]] + Var [ **E** [ *X* *|* *Y* ]] . by the definition of variance (1.45)

*1.1.11* *Change of Variables*

It is often useful to understand the distribution of a transformed ran
dom variable *Y* = *g* ( *X* ) that is defined in terms of a random variable
*X*, whose distribution is known. Let us first consider the univariate
setting. We would like to express the distribution of *Y* in terms of the
distribution of *X*, that is, we would like to find

*P* *Y* ( *y* ) = **P** ( *Y* *≤* *y* ) = **P** ( *g* ( *X* ) *≤* *y* ) = **P** *X* *≤* *g* *[−]* [1] ( *y* ) . (1.51)
� �

When the random variables are continuous, this probability can be expressed as an integration over the domain of *X* . We can then use the
substitution rule of integration to “change the variables” to an integration over the domain of *Y* . Taking the derivative yields the density
*p* *Y* . [16] 16 The full proof of the change of variables formula in the univariate setting
can be found in section 6.7.2 of “Mathematics for machine learning” (Deisenroth et al.).

18 probabilistic artificial intelligence

There is an analogous change of variables formula for the multivariate
setting.


**Jacobian** Given a vector-valued function,




,






***g*** : **R** *[n]* *→* **R** *[m]*, ***x*** *�→*







*g* 1 ( ***x*** )

...
*g* *m* ( ***x*** )



where *g* *i* : **R** *[n]* *→* **R**, the *Jacobian* of ***g*** at
***x*** *∈* **R** *[n]* is


*∂∂* *g* ***x*** 1 ( ( 1 ***x*** ) ) *· · ·* *∂∂* *g* ***x*** 1 ( ( *n* ***x*** ) )


*∂* *g* 1 ( ***x*** )


*∂* ***x*** ( 1 ) *∂* ***x*** ( *n* )

... ... ...
*∂∂* *g* ***x*** *m* ( ( 1 ***x*** ) ) *· · ·* *∂∂* *g* ***x*** *m* ( ( *n* ***x*** ) )


*∂* ***x*** ( *n* )


Here, the term ��det� ***Dg*** *[−]* [1] ( ***y*** ) ��� measures how much a unit volume
changes when applying ***g*** . Intuitively, this swaps the coordinate system over which we integrate. The factor ��det� ***Dg*** *[−]* [1] ( ***y*** ) ��� corrects for
the change in volume that is caused by this change in coordinates.

*1.2*
*Supervised Learning*

In *supervised learning*, we want to learn a function *f* : *X →Y* from
labeled training data. That is, we are given a collection of labeled
examples, *D* = . *{* ( ***x*** *i*, *y* *i* ) *}* *in* = 1 [, where] ***[ x]*** *[i]* *[ ∈X]* [ are] *[ inputs]* [ and the] *[ y]* *[i]* *[ ∈Y]*
are *labels*, and we want to find a function *f* [ˆ] that best-approximates *f* .
It is common to consider a class of functions *F* (called the *function*
*class* ) that serve as candidates for *f* [ˆ], where each function is described
by some parameters ***θ*** .

It is often assumed that the observations are noisy, that is,

*y* *i* = *f* ( ***x*** *i* ) + *ϵ* *i* ( ***x*** *i* ), (1.55)

where *ϵ* *i* ( ***x*** *i* ) is some independent zero-mean noise, for example Gaussian. Often, the dependence of the noise on ***x*** *i* is omitted for brevity.
When the noise distribution may depend on ***x*** *i*, the noise is said to be

*heteroscedastic* and otherwise the noise is called *homoscedastic* . We will

describe this model in greater detail when introducing Bayesian linear
regression in chapter 2.

Typically, we differentiate between the task of *regression* where *Y* = . **R** *k*

(often, we learn a scalar label, so *k* = 1), and the task of *classification*
where *Y* = . *C* and *C* is an *m* -element set of classes. In other words,

regression is the task of predicting a continuous label, whereas classification is the task of predicting a discrete class label. These two tasks
are intimately related: in fact, we can think of classification tasks as
a regression problem where we learn a probability distribution over
class labels. In this regression problem, *Y* = . ∆ *C* where we defined ∆ *C*




.



***Dg*** ( ***x*** ) = .







(1.53)

Observe that for a function *f* : **R** *[n]* *→* **R**,

***D*** *f* ( ***x*** ) = ***∇*** *f* ( ***x*** ) *[⊤]* . (1.54)

fundamentals 19


as the set of all probability distributions over the set of classes *C* . Recall
from remark 1.10 that ∆ *[C]* is an ( *m* *−* 1 ) -dimensional convex polytope
in the space of probabilities [ 0, 1 ] *[m]* .

*1.2.1* *Empirical Risk and Population Risk*

We denote the underlying (and unknown) joint distribution over inputlabel pairs ( ***x***, *y* ) by *P* . The notion of “approximation error” is typically captured by a loss function *ℓ* ( ˆ *y* ; *y* ) *∈* **R** which is small when the
prediction ˆ *y* is “close” to the true label *y* and large otherwise. As our
objective is to best-approximate the mappings ( ***x***, *y* ) *∼P*, we aim to

minimize

**E** ( ***x***, *y* ) *∼P* � *ℓ* ( *f* [ˆ] ( ***x*** ) ; *y* ) �. (1.56)

This quantity is also called the *population risk* . However, the underlying

distribution *P* is unknown to us. All that we can work with is the
training data for which we assume *D* [iid] *∼P* . It is therefore natural to
consider minimizing


1

*n*


*n*
###### ∑ ℓ ( f [ˆ] ( x i ) ; y i ), D = { ( x i, y i ) } i [n] = 1 [,] (1.57)

*i* = 1


which is known as the *empirical risk* .

However, selecting *f* [ˆ] by minimizing the empirical risk can be problematic. The reason is that in this case the model *f* [ˆ] and the empirical risk
depend on the same data *D*, implying that the empirical risk will not
be an unbiased estimator of the population risk. [17] This can result in a 17 In section 1.4.1, we discuss estimator
model which fits the training data too closely (called *overfitting* ), and bias in detail.
which is failing to generalize to unseen data. We will discuss some
solutions to this problem in section 4.4 when covering model selection
in the context of Gaussian processes.

*1.3* *Bayesian Learning and Inference*

Bayesian learning is the process of updating a Bayesian prior **P** ( *A* ) to
a Bayesian posterior **P** ( *A* *|* *B* ) upon observing *B* . Thereby, we typically
replace *A* with the parameterization of a model ***θ***, and *B* with labeled
training data *{* ( ***x*** *i*, *y* *i* ) *}* *i* *[n]* = 1 [.]

At this point, it is helpful to differentiate between learning and inference. By *learning* (or estimation) we refer to the aforementioned
process of learning a model from data. In contrast, by *inference* (or prediction) we refer to the process of using our learned model to predict
labels at new inputs. [18] 18 In the literature, these terms are sometimes used differently, where our learnAt the core of Bayesian learning is Bayes’ rule. ing is referred to as “inference” and our
inference is referred to as “prediction”.

20 probabilistic artificial intelligence



*Proof.* Bayes’ rule is a direct consequence of the definition of conditional densities (1.13) and the product rule (1.14).



It is useful to consider the meaning of each term separately:

- *p* ( ***x*** *|* ***y*** ) ( *posterior* ) is the updated belief about ***x*** after observing ***y***,

- *p* ( ***x*** ) ( *prior* ) is the initial belief about ***x***,

- *p* ( ***y*** *|* ***x*** ) ( *(conditional) likelihood* ) describes how likely the observations ***y*** are under a given value ***x***,

- *p* ( ***x***, ***y*** ) = *p* ( ***y*** *|* ***x*** ) *p* ( ***x*** ) ( *joint likelihood* or *generative model* ) combines
prior and likelihood,

- *p* ( ***y*** ) ( *evidence* or *marginal likelihood* ) describes how likely the observations ***y*** are across all values of ***x*** .

The evidence can be computed using the law of total probability (1.15),


*p* ( ***y*** ) =
�


**X** ( Ω ) *[p]* [(] ***[y]*** *[ |]* ***[ x]*** [)] *[p]* [(] ***[x]*** [)] *[ d]* ***[x]*** [,] (1.59)


or the sum rule. Note, however, that as the evidence does not depend

on ***x***,

*p* ( ***x*** *|* ***y*** ) ∝ *p* ( ***y*** *|* ***x*** ) *p* ( ***y*** ), (1.60)

so we do not need to compute it to optimize for ***x*** (e.g., find the gradient with respect to ***x*** ). This is a helpful fact, which we will make use
of quite often.


fundamentals 21

*1.3.1* *Using Bayes’ rule for supervised learning*

Returning to the task of learning a model from training data, we interpret *p* ( ***θ*** ) as the degree of our belief that the model parameterized
by ***θ*** “describes the data best”. The likelihood describes how likely the
training data is under a particular model. Often, we assume that our
data was sampled independently, allowing us to write the likelihood
as a product:

*n*
###### p ( y 1: n | x 1: n, θ ) = ∏ p ( y i | x i, θ ) . (1.61)

*i* = 1

The posterior represents our belief about the best model after seeing
the training data. Using Bayes’ rule, we can write it as [19] 19 We generally assume that

###### p ( θ | x 1: n, y 1: n ) = [1] ∏ n p ( y i | x i, θ ) where (1.62)

*Z* *[p]* [(] ***[θ]*** [)] *i* = 1


*p* ( ***θ*** *|* ***x*** 1: *n* ) = *p* ( ***θ*** ) .

For our purposes, you can think of the
inputs ***x*** 1: *n* as fixed deterministic param
eters.


*n*
###### Z = . p ( θ ) ∏ p ( y i | x i, θ ) d θ (1.63)
� *i* = 1

and again assuming that the training data was sampled independently.
We often refer to *Z* as the *normalizing constant* .

Finally, we can use our learned model for inference (predictions) at a
new input ***x*** *[⋆]* by conditioning on ***θ***,


*p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ) = *p* ( *y* *[⋆]*, ***θ*** *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ) *d* ***θ*** by the sum rule (1.10)
�

= *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ) *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) *d* ***θ*** . (1.64) by the product rule (1.14) and
�
*y* *[⋆]* *⊥* ***x*** 1: *n*, *y* 1: *n* *|* ***θ***

Here, the distribution over models *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) is called the *posterior*
and the distribution over predictions *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ) is called the
*predictive posterior* .

*1.3.2* *Conjugate Priors*

In general, the integrals in the normalizing constant *Z* (1.63) and the
predictive distribution (1.64) cannot be expressed in closed-form.

If the prior *p* ( ***x*** ) and posterior *p* ( ***x*** *|* ***y*** ) are of the same family of
distributions, the prior is called a *conjugate prior* to the likelihood *p* ( ***y*** *|*
***x*** ) . This is a very desirable property, as it allows us to apply the same
learning mechanism recursively.

22 probabilistic artificial intelligence

Under some conditions the Gaussian is self-conjugate (cf. section 2.3.1).
That is, if we have a Gaussian prior and a Gaussian likelihood, then
our posterior will also be a Gaussian.










*1.4* *Parameter Estimation*

A very common — albeit non-Bayesian — approach to learning is to
reduce the posterior distribution *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) to a point estimate of
***θ*** . We will now take a brief look at this approach, but first define the
concept of an estimator more formally.

*1.4.1* *Estimators*

Suppose we are given a collection of independent samples ***x*** [(] [1] [)], . . ., ***x*** [(] *[n]* [)]

from some random vector **X** . Often, the exact distribution of **X** is un
known to us, but we still want to “estimate” some property of this
distribution, for example its mean. We denote the property that we
aim to estimate from our sample by ***θ*** . For example, if our goal is
estimating the mean of **X**, then ***θ*** = [.] **E** [ **X** ] .

An *estimator* for a parameter ***θ*** is a random vector **U** that is a function
of *n* sample variables **X** [(] [1] [)], . . ., **X** [(] *[n]* [)] whose distribution is identical to
the distribution of **X** . Any concrete sample ***x*** [(] *[i]* [)] *∼* **X** [(] *[i]* [)] yields a con
fundamentals 23

crete estimate ***u*** *∼* **U** of ***θ*** . How these samples are composed is not
restricted, but their composition should lead to a “good estimate” of
***θ*** . Mainly, there are two measures of goodness of an estimator: its
expectation and its variance.

Clearly, we want **E** [ **U** ] = ***θ*** . Estimators that satisfy this property are
called *unbiased* . The *bias*, **E** **U** [ **U** *−* ***θ*** ], of an unbiased estimator is **0** .

Moreover, we want that the variance of *U* is small. [20] A common mea- 20 The variance of estimators **U** is studied
sure for the variance of an estimator of *θ* is the *mean squared error*, component wise.

MSE ( *U* ) = . **E** *U* � ( *U* *−* *θ* ) [2] [�] . (1.67)

We say that an estimator is *consistent* if its mean squared error con
verges to zero as *n* *→* ∞.






24 probabilistic artificial intelligence










covered using all other samples and the
sample mean.



We further say that an estimator *U* is *sharply concentrated* around *θ* if
for any *ϵ* *>* 0,

.
**P** ( *|* *U* *−* *θ* *| >* *ϵ* ) *≤* exp ( *−* Ω ( *ϵ* )) = *δ*, (1.72)


where Ω ( *ϵ* ) denotes the class of functions that grow at least linearly in
*ϵ* . [22] Thus, if an estimator is sharply concentrated, its absolute error is 22 That is, *h* *∈* Ω ( *ϵ* ) iff lim *ϵ* *→* ∞ *h* ( *ϵϵ* )


bounded by an exponentially quickly decaying error probability *δ* as *ϵ*

grows.


22 That is, *h* *∈* Ω ( *ϵ* ) iff lim *ϵ* *→* ∞ *ϵ* *>* 0.

With slight abuse of notation, we force
*h* to be positive (so as to ensure that
the argument to the exponential function is negative) whereas in the traditional definition of Landau symbols, *h* is
only required to grow linearly in absolute value.



*p* ( *x* )


fundamentals 25

5 10

*x*





10 *[−]* [1]

10 *[−]* [3]

10 *[−]* [5]

10 *[−]* [7]











Figure 1.5: Shown are the right tails of
the PDFs of a **Gaussian** with mean 1 and

variance 1, a **exponential distribution**
with mean 1 and parameter *λ* = 1, and
a **log-normal distribution** with mean 1
and variance 1 on a log-scale.

*p* ( *x* )

0.4

0.2

0.0

0 2 4

*x*

Figure 1.6: Shown are the right tails of
the PDFs of a **Laplace distribution** with
mean 0 and length scale 1 and a **Gaus-**
**sian** with mean 0 and variance 1.



*1.4.2* *Mean Estimation and Concentration Inequalities*

We have seen that we desire two properties in estimators: namely,
(1) that they are unbiased; and (2) that their variance is small. [23] For 23 That is, they are consistent and, ideestimating expectations with the sample mean *X* *n* which is also known ally, their variance converges quickly.
as a *Monte Carlo approximation*, we will see that both properties follow
from standard results in probability theory.

It is immediate from the definition of the sample mean (1.69) that it
is an unbiased estimator for **E** [ *X* ] . We will now see that the sample
mean is also consistent, and even better, converges with exponentially
decaying error probability.

26 probabilistic artificial intelligence

First, let us recall the common notions of convergence of a sequence of
random variables *X* *n* .

**Definition 1.43** (Convergence of random variables) **.** Let *{* *X* *n* *}* *n* *∈* **N** be a
sequence of random variables and *X* another random variable. We say
that,

1. *X* *n* converges to *X almost surely* (also called *convergence with proba-*
*bility* 1 or *equality with high probability* ) if

**P** *ω* *∈* Ω : lim = 1, (1.77)
�� *n* *→* ∞ *[X]* *[n]* [(] *[ω]* [) =] *[ X]* [(] *[ω]* [)] ��

a.s.
and we write *X* *n* *→* *X* as *n* *→* ∞.

2. *X* *n* converges to *X in probability* if for any *ϵ* *>* 0,

lim (1.78)
*n* *→* ∞ **[P]** [(] *[|]* *[X]* *[n]* *[ −]* *[X]* *[|][ >]* *[ ϵ]* [) =] [ 0,]

**P**
and we write *X* *n* *→* *X* as *n* *→* ∞.
3. *X* *n* converges to *X in distribution* if for all points *x* *∈* *X* ( Ω ) at which
*P* *X* is continuous,

lim (1.79)
*n* *→* ∞ *[P]* *[X]* *[n]* [(] *[x]* [) =] *[ P]* *[X]* [(] *[x]* [)] [,]

*D*
and we write *X* *n* *→* *X* as *n* *→* ∞.

It can be shown that as *n* *→* ∞,

a.s. **P** *D*
*X* *n* *→* *X* = *⇒* *X* *n* *→* *X* = *⇒* *X* *n* *→* *X* . (1.80)

In the following, we will use some classical concentration inequalities.




The law of large numbers is a classical result in statistics and directly
implies that the sample mean is consistent.

fundamentals 27







Under more restrictive assumptions it is even possible to show that the
sample mean *X* *n* is sharply concentrated around the mean **E** *X*, which
is a much stronger property than consistency. One such assumption is
provided by the notion of a sub-Gaussian random variable.

**Definition 1.47** (Sub-Gaussian random variable) **.** A random variable
*X* : Ω *→* **R** is called *σ* - *sub-Gaussian* for *σ* *>* 0 if for all *λ* *∈* **R**,


2 2
*σ* *λ*
**E** *e* *[λ]* [(] *[X]* *[−]* **[E]** *[X]* [)] [�] *≤* exp
� � 2


. (1.85)
�





28 probabilistic artificial intelligence





In words, the absolute error of the expectation estimate is bounded by
an exponentially quickly decaying error probability *δ* . Solving for *n*,

we obtain that for


*n* *≥* [2] *[σ]* [2]


(1.88)
*δ*



*[σ]*

*ϵ* [2] [ log 2] *δ*


the probability that the absolute error is greater than *ϵ* is at most *δ* .

*Proof.* Let *S* *n* = . *nX* *n* = *X* 1 + *· · ·* + *X* *n* . We have for any *λ*, *ϵ* *>* 0 that


**P** �


*X* *n* *−* **E** *X* *≥* *ϵ* � = **P** ( *S* *n* *−* **E** *S* *n* *≥* *nϵ* )

= **P** *e* *[λ]* [(] *[S]* *[n]* *[−]* **[E]** *[S]* *[n]* [)] *≥* *e* *[n][ϵλ]* [�] using that *z* *�→* *e* *[λ][z]* is increasing
�

*≤* *e* *[−]* *[n][ϵλ]* **E** [ *e* *[λ]* [(] *[S]* *[n]* *[−]* **[E]** *[S]* *[n]* [)] ] using Markov’s inequality (1.81)

*n*
###### = e [−] [n][ϵλ] ∏ E [ e [λ] [(] [X] [i] [−] [E] [X] [)] ] using independence of the X i

*i* = 1

*n*
###### ≤ e [−] [n][ϵλ] ∏ e [σ] [2] [λ] [2] [/2] using the characterizing property of a

*i* = 1 *σ* -sub-Gaussian random variable (1.85)


= exp *−* *nϵλ* + *[n][σ]* [2] *[λ]* [2]
� 2


.
�


Minimizing the expression with respect to *λ*, we set *λ* = *ϵ* / *σ* [2], and

obtain


.
�


**P** � *X* *n* *−* **E** *X* *≥* *ϵ* � *≤* min

*λ* *>* 0


exp *−* *nϵλ* + *[n][σ]* [2] *[λ]* [2] = exp *−* *[n][ϵ]* [2]
� � 2 �� � 2 *σ* [2]


The theorem then follows from


**P** ��� *X* *n* *−* **E** *X* �� *≥* *ϵ* � = **P** �


*X* *n* *−* **E** *X* *≥* *ϵ* � + **P** �


*X* *n* *−* **E** *X* *≤−* *ϵ* �


and noting that the second term can be bounded analogously to the
first term by considering the random variables *−* *X* 1, . . ., *−* *X* *n* .

The law of large numbers and Hoeffding’s inequality show that we
can estimate **E** [ *f* ( *X* )] very precisely with “few” samples using a sample mean. Crucially, we require that the samples *x* *i* are from the same
distribution as the random variable *X* and that the samples are independent.

fundamentals 29


*1.4.3* *Maximum Likelihood Estimation*

Let us return to finding point estimates of our model. Intuitively, the
“best model” should have the property that the training data is as
likely as possible under this model when compared to all other models. This is precisely the *maximum likelihood estimate* (or MLE):

***θ*** ˆ MLE = . arg max *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) (1.89)
***θ***


= arg max
***θ***

= arg max
***θ***


*n*
###### ∏ p ( y i | x i, θ ) using the independence of the training

*i* = 1 data (1.61)

*n*
###### ∑ log p ( y i | x i, θ ) . (1.90) using that log is a monotonic function

*i* = 1


Taking the logarithm of the likelihood is a standard technique. The
resulting log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) is called *log-likelihood*, and its negative is
known as the *negative log-likelihood* *ℓ* nll ( ***θ*** ; *D* ),

= arg min *ℓ* nll ( ***θ*** ; *D* ) . (1.91)
***θ***










30 probabilistic artificial intelligence

*1.4.4* *Maximum A Posteriori Estimation*

Often, the MLE overfits to the training data. The danger of overfitting
can be reduced by taking a “more Bayesian” approach and maximizing
over the entire posterior distribution instead of only maximizing the
likelihood. This results in the *maximum a posteriori estimate* (or MAP
estimate):

***θ*** ˆ MAP = . arg max *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) (1.93)
***θ***

= arg max *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) *·* *p* ( ***θ*** ) (1.94) by Bayes’ rule (1.58)
***θ***

*n*
###### = arg max log p ( θ ) + ∑ log p ( y i | x i, θ ) . (1.95) using that log is a monotonic function

***θ*** *i* = 1


= arg min *−* log *p* ( ***θ*** )
***θ*** � �� �
regularization


+ *ℓ* nll ( ***θ*** ; *D* )
� �� �
quality of fit


. (1.96)


Here, the *log-prior* log *p* ( ***θ*** ) acts as a regularizer. For example,

*σ* [2]

- if *p* ( ***θ*** ) = *N* ( ***θ*** ; **0**, *σ* *p* [2] ***[I]*** [)] [ then] *[ −]* [log] *[ p]* [(] ***[θ]*** [) =] 2 *p* *[∥]* ***[θ]*** *[∥]* 2 [2] [,]

- if *p* ( ***θ*** ) = Laplace ( ***θ*** ; **0**, *h* ) then *−* log *p* ( ***θ*** ) = 2 *[h]* *[∥]* ***[θ]*** *[∥]* 1 [2] [, and]

- for a uniform prior (i.e., a prior that is independent of ***θ*** ), MAP

estimation reduces to likelihood maximization.

Note that in case the posterior is Gaussian, the MAP estimate (i.e.,
mode of the posterior distribution) corresponds to the mean.

*1.5* *Optimization*


*θ* 2

1.0

0.5

0.0

*−* 0.5

*−* 1.0

*−* 1 0 1

*θ* 1

Figure 1.7: Level sets of **Gaussian** and
**Laplace** regularization. It can be seen
that Laplace regularization is more effective in encouraging sparse solutions
(that is, solutions where many components are set to exactly 0).


Finding parameter estimates is one of the many examples where we
seek to minimize some function *ℓ* . [24] The field of optimization has a 24 W.l.o.g. we assume that we want to

rich history, which we will not explore in much detail here. What will minimize *ℓ* . If we wanted to maximize

the objective, we can simply minimize its

be important for us is that given that the function to be optimized negation.
(called the *objective* or *loss* ) fulfills certain smoothness properties, optimization is a well-understood problem and can often be solved exactly
(e.g., when the objective is convex) or “approximately” when the objective is non-convex. In fact, we will see that it is often advantageous
to frame problems as optimization problems when suitable because
the machinery to solve these problems is so extensive.


*1.5.1* *Stationary Points*

In this section, we derive some basic facts about unconstrained optimization problems. Given some function *f* : **R** *[n]* *→* **R**, we want to
find

min (1.97)
***x*** *∈* **R** *[n]* *[ f]* [ (] ***[x]*** [)] [.]

We say that a point ***x*** *[⋆]* *∈* **R** *[n]* is a *(global) optimum* of *f* if *f* ( ***x*** *[⋆]* ) *≤* *f* ( ***x*** )
for any ***x*** *∈* **R** *[n]* .

Consider the more general problem of minimizing *f* over some subset
*S* *⊆* **R** *[n]*, that is, to minimize the function *f* *S* : *S* *→* **R**, ***x*** *�→* *f* ( ***x*** ) . If there
exists some open subset *S* *⊆* **R** *[n]* including ***x*** *[⋆]* such that ***x*** *[⋆]* is optimal
with respect to the function *f* *S*, then ***x*** *[⋆]* is called a *local optimum* of *f* .



*x* [3]


fundamentals 31

*−* 2 0 2

*x*


**Definition 1.52** (Stationary point) **.** Given a function *f* : **R** *[n]* *→* **R**, a
point ***x*** *∈* **R** *[n]* where ***∇*** *f* ( ***x*** ) = **0** is called a *stationary point* of *f* .

Being a stationary point is not sufficient for optimality. Take for example the point *x* = . 0 of *f* ( *x* ) = . *x* 3 . Such a point that is stationary but
not (locally) optimal is called a *saddle point* .

**Theorem 1.53** (First-order optimality condition) **.** *If* ***x*** *∈* **R** *[n]* *is a local*
*extremum of a differentiable function f* : **R** *[n]* *→* **R** *, then* ***∇*** *f* ( ***x*** ) = **0** *.*

*Proof.* Assume ***x*** is a local minimum of *f* . Then, for all ***d*** *∈* **R** *[n]* and for
all small enough *λ* *∈* **R**, we have *f* ( ***x*** ) *≤* *f* ( ***x*** + *λ* ***d*** ), so

0 *≤* *f* ( ***x*** + *λ* ***d*** ) *−* *f* ( ***x*** )


20

0

*−* 20


Figure 1.8: Example of a saddle point at
*x* = 0.


= *λ* ***∇*** *f* ( ***x*** ) *[⊤]* ***d*** + *o* ( *λ* *∥* ***d*** *∥* 2 ) . using a first-order expansion of *f*
around ***x***
Dividing by *λ* and taking the limit *λ* *→* 0, we obtain

0 *≤* ***∇*** *f* ( ***x*** ) *[⊤]* ***d*** + lim *o* ( *λ* *∥* ***d*** *∥* 2 ) = ***∇*** *f* ( ***x*** ) *[⊤]* ***d*** .
*λ* *→* 0 *λ*

. 2
Take ***d*** = *−* ***∇*** *f* ( ***x*** ) . Then, 0 *≤−∥* ***∇*** *f* ( ***x*** ) *∥* 2 [, so] ***[ ∇]*** *[f]* [ (] ***[x]*** [) =] **[ 0]** [.]

32 probabilistic artificial intelligence

*1.5.2* *Convexity*

Convex functions are a subclass of functions where finding global minima is substantially easier than for general functions.

**Definition 1.54** (Convex function) **.** A function *f* : **R** *[n]* *→* **R** is *convex* iff

*∀* ***x***, ***y*** *∈* **R** *[n]* : *∀* *θ* *∈* [ 0, 1 ] : *f* ( *θ* ***x*** + ( 1 *−* *θ* ) ***y*** ) *≤* *θ f* ( ***x*** ) + ( 1 *−* *θ* ) *f* ( ***y*** ) .
(1.99)

That is, any line between two points on *f* lies “above” *f* . If the inequality of eq. (1.99) is strict, we say that *f* is *strictly convex* .

If the function *f* is convex, we say that the function *−* *f* is *concave* . The
above is also known as the 0 *th-order characterization of convexity* .

**Theorem 1.55** (First-order characterization of convexity) **.** *Suppose that*
*f* : **R** *[n]* *→* **R** *is differentiable, then f is convex iff*

*∀* ***x***, ***y*** *∈* **R** *[n]* : *f* ( ***y*** ) *≥* *f* ( ***x*** ) + ***∇*** *f* ( ***x*** ) *[⊤]* ( ***y*** *−* ***x*** ) . (1.100)

Observe that the right-hand side of the inequality is an affine function
with slope ***∇*** *f* ( ***x*** ) based at *f* ( ***x*** ) .

*Proof.* In the following, we will make use of directional derivatives.

|Col1|f|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||

|Col1|f|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||


*−* 2 0 2

*x*

Figure 1.10: The first-order characterization characterizes convexity in terms of
affine lower bounds. Shown is an affine
lower bound based at *x* = *−* 2.


*y*


8

6

4

2

0



*−* 2 0 2

*x*

Figure 1.9: Example of a convex function. Any line between two points on *f*
lies “above” *f* .

*y*


8

6

4

2

0








- “ *⇒* ”: Fix any ***x***, ***y*** *∈* **R** *[n]* . As *f* is convex,

*f* (( 1 *−* *θ* ) ***x*** + *θ* ***y*** ) *≤* ( 1 *−* *θ* ) *f* ( ***x*** ) + *θ f* ( ***y*** ),

for all *θ* *∈* [ 0, 1 ] . We can rearrange to


*f* (( 1 *−* *θ* ) ***x*** + *θ* ***y***
� �� �
***x*** + *θ* ( ***y*** *−* ***x*** )


) *−* *f* ( ***x*** ) *≤* *θ* ( *f* ( ***y*** ) *−* *f* ( ***x*** )) .

fundamentals 33


Dividing by *θ* yields,

*f* ( ***x*** + *θ* ( ***y*** *−* ***x*** )) *−* *f* ( ***x*** )

*≤* *f* ( ***y*** ) *−* *f* ( ***x*** ) .
*θ*

Taking the limit *θ* *→* 0 on both sides gives the directional derivative
at ***x*** in direction ***y*** *−* ***x***,

***∇*** *f* ( ***x*** ) *[⊤]* ( ***y*** *−* ***x*** ) = *D f* ( ***x*** )[ ***y*** *−* ***x*** ] *≤* *f* ( ***y*** ) *−* *f* ( ***x*** ) .

- “ *⇐* ”: Fix any ***x***, ***y*** *∈* **R** *[n]* and let ***z*** = . *θ* ***y*** + ( 1 *−* *θ* ) ***x*** . We have,

*f* ( ***y*** ) *≥* *f* ( ***z*** ) + ***∇*** *f* ( ***z*** ) *[⊤]* ( ***y*** *−* ***z*** ), and

*f* ( ***x*** ) *≥* *f* ( ***z*** ) + ***∇*** *f* ( ***z*** ) *[⊤]* ( ***x*** *−* ***z*** ) .

We also have ***y*** *−* ***z*** = ( 1 *−* *θ* )( ***y*** *−* ***x*** ) and ***x*** *−* ***z*** = *θ* ( ***x*** *−* ***y*** ) . Hence,


*θ f* ( ***y*** ) + ( 1 *−* *θ* ) *f* ( ***x*** ) *≥* *f* ( ***z*** ) + ***∇*** *f* ( ***z*** ) *[⊤]* ( *θ* ( ***y*** *−* ***z*** ) + ( 1 *−* *θ* )( ***x*** *−* ***z*** )
� �� �
**0**

= *f* ( *θ* ***y*** + ( 1 *−* *θ* ) ***x*** ) .


)



*Proof.* By the first-order characterization of convexity, we have for any
***y*** *∈* **R** *[n]*,


*f* ( ***y*** ) *≥* *f* ( ***x*** ) + ***∇*** *f* ( ***x*** ) *[⊤]*
� �� �
**0**


( ***y*** *−* ***x*** ) = *f* ( ***x*** ) .


Generally, the main difficulty in solving convex optimization problems
lies therefore in finding stationary points (or points that are sufficiently
close to stationary points).


34 probabilistic artificial intelligence



**Hessian** The *Hessian* of a function *f* :
**R** *[n]* *→* **R** at a point ***x*** *∈* **R** *[n]* is

***H*** *f* ( ***x*** ) = . ***H*** *f* ( ***x*** )


*∂* ***x*** *∂* ( 1 [2] ) *f* *∂* ( ***xx*** ) ( 1 ) *· · ·* *∂* ***x*** *∂* ( 1 [2] ) *f* *∂* ( ***xx*** ) ( *n* )

... ... ...
*∂* ***x*** *∂* ( *n* [2] ) *f* *∂* ( ***xx*** ) ( 1 ) *· · ·* *∂* ***x*** ( *∂n* [2] ) *f* *∂* ( ***xx*** ) ( *n* )







= .







*1.5.3* *Stochastic Gradient Descent*

In this course, we primarily employ so-called first-order methods, which
rely on (estimates of) the gradient of the objective function to determine a direction of local improvement. The main idea behind these
methods is to repeatedly take a step in the opposite direction of the
gradient scaled by a learning rate *η* *t*, which may depend on the cur
rent iteration *t* .

We will often want to minimize a stochastic optimization objective

*L* ( ***θ*** ) = . **E** ***x*** *∼* *p* [ *ℓ* ( ***θ*** ; ***x*** )] (1.106)

where *ℓ* and its gradient ***∇*** *ℓ* are known.

Based on our discussion in previous subsections, it is a natural first
step to look for stationary points of *L*, that is, the roots of ***∇*** *L* .


(1.104)

= ( ***D*** ***∇*** *f* ( ***x*** )) *[⊤]* (1.105)

*∈* **R** *[n]* *[×]* *[n]* .

Thus, a Hessian captures the curvature
of *f* . If the Hessian of *f* is positive definite at a point ***x***, then *f* is “curved up
around ***x*** ”.
Hessians are symmetric when the second partial derivatives are continuous,
due to Schwartz’s theorem.




learning and stochastic approximations”




sure convergence of a sequence of random variables.

fundamentals 35


*η* *t* = [1] / *t* is an example of a learning rate satisfying the RM-conditions.

Using Robbins-Monro to find a root of ***∇*** *L* is known as *stochastic gradi-*
*ent descent* . In particular, when *ℓ* is convex and the RM-conditions are
satisfied, Robbins-Monro converges to a stationary point (and hence, a
global minimum) of *L* almost surely.

Moreover, it can be shown that for general *ℓ* and satisfied RM-conditions,
stochastic gradient descent converges to a local minimum almost surely.
Intuitively, the randomness in the gradient estimates allows the algorithm to “jump past” saddle points.

A commonly used strategy to obtain unbiased gradient estimates is
to take the sample mean of the gradient with respect to some set of
samples *B* (also called a *batch* ) as is shown in alg. 1.61.

**Al** **g** **orithm 1.61:** Stochastic g radient descent, SGD

**1** initialize ***θ***

**2** `repeat`

**3** draw mini-batch *B* = . *{* ***x*** ( 1 ), . . ., ***x*** ( *m* ) *}*, ***x*** ( *i* ) *∼* *p*

**4** ***θ*** *←* ***θ*** *−* *η* *t* *m* [1] [∑] *i* *[m]* = 1 ***[∇]*** ***[θ]*** *[ ℓ]* [(] ***[θ]*** [;] ***[ x]*** [(] *[i]* [)] [)]

**5** `until` `converged`





36 probabilistic artificial intelligence








We will have a brief look at other approaches to stochastic optimization
in sections 6.3 and 6.4.

*1.6* *Multivariate Normal Distribution*

Using general distributions for learning and inference is computationally very expensive when the number of dimensions is large — even
in the discrete setting. Computing marginal distributions using the
sum rule yields an exponentially long sum in the size of the random
vector. Similarly, the normalizing constant of the conditional distribution is a sum of exponential length. Even to represent any discrete
joint probability distribution requires space that is exponential in the
number of dimensions (cf. fig. 1.11). One strategy to get around this
computational blowup is to restrict the class of distributions.

It turns out that Gaussians have many extremely useful properties:
they have a compact representation and — as we will see in later chapters — they allow for closed-form learning and inference.

In eq. (1.6), we have already seen the PDF of the univariate Gaussian distribution. A random vector **X** in **R** *[n]* is *normally distributed*,
**X** *∼N* ( ***µ***, ***Σ*** ), if its PDF is


fundamentals 37

*X* 1 *· · ·* *X* *n* *−* 1 *X* *n* **P** ( *X* 1: *n* )
0 *· · ·* 0 0 0.01

0 *· · ·* 0 1 0.001

0 *· · ·* 1 0 0.213

*· · ·* *· · ·* *· · ·* *· · ·*

1 *· · ·* 1 1 0.0003

Figure 1.11: A table representing a joint
distribution of *n* binary random variables. The number of parameters is
2 *[n]* *−* 1.

0.15

0.10

0.05







0.3

0.2

0.1

0.0






where ***µ*** *∈* **R** *[n]* is the mean, ***Σ*** *∈* **R** *[n]* *[×]* *[n]* the covariance matrix, and ***Λ*** = .
***Σ*** *[−]* [1] the so-called *precision matrix* . *N* ( **0**, ***I*** ) is the multivariate *standard*
*normal distribution* . In this definition, we assume that the covariance
matrix ***Σ*** is invertible, i.e., does not have the eigenvalue 0.


Figure 1.12: Shown are two-dimensional
Gaussians with mean **0** and covariance

matrices



*x*

*x*




2

2




27 An *affine transformation* is a function ***f*** :
**R** *[n]* *→* **R** *[m]*, ***x*** *�→* ***Ax*** + ***b*** where ***A*** *∈* **R** *[m]* *[×]* *[n]*

is an invertible linear map and ***b*** *∈* **R** *[m]*


***Σ*** 1 = . �10 01

respectively.


�, ***Σ*** 2 = . �0.91 0.91


�


As we have already seen that a covariance matrix does not have nega- is called a translation.
tive eigenvalues, [28] this ensures that ***Σ*** and ***Λ*** are positive definite. [29] 28 see exercise 1.30


is called a translation.


29 The inverse of a positive definite maWe call a Gaussian *isotropic* iff its covariance matrix is of the form trix is also positive definite.
***Σ*** = *σ* [2] ***I*** for some *σ* [2] *∈* **R** . In this case, the sublevel sets of the PDF are

perfect spheres.

Note that a Gaussian can be represented using only *O* � *n* [2] [�] parameters. In the case of a diagonal covariance matrix (which as we will see
corresponds to independent univariate Gaussians) we just need *O* ( *n* )

parameters.

38 probabilistic artificial intelligence



*1.6.1* *Gaussian Random Vectors*

**Definition 1.67** (Gaussian random vector, GRV) **.** Given random variables *X* 1, . . ., *X* *n*, the random vector **X** = [ *X* 1, . . ., *X* *n* ] in **R** *[n]* is called a
*Gaussian random vector* iff any linear combination of its coordinates is
normally distributed.

The definition immediately yields the following consequences:

- Any marginal of a GRV is a GRV.

- Any affine transformation of a GRV is a GRV.

**Theorem 1.68.** *Two jointly Gaussian random vectors,* **X** *and* **Y** *, are indepen-*
*dent if and only if* **X** *and* **Y** *are uncorrelated.*

*Proof.* Recall that independence of any random vectors **X** and **Y** implies that they are uncorrelated, [30] the converse implication is a special 30 This is because the expectation of their
property of Gaussian random vectors. product factorizes, see eq. (1.26).

Consider the joint Gaussian random vector **Z** = [ . **X Y** ] *∼N* ( ***µ***, ***Σ*** ) and
assume that **X** *∼N* ( ***µ*** **X**, ***Σ*** **X** ) and **Y** *∼N* ( ***µ*** **Y**, ***Σ*** **Y** ) are uncorrelated.
Then, ***Σ*** can be expressed as


***Σ*** =


***Σ*** **X** **0**

**0** ***Σ*** **Y**
� �


,


implying that the PDF of **Z** factorizes,

*N* ([ ***x y*** ] ; ***µ***, ***Σ*** )


1

*−* [1]
exp
det ( 2 *π* ***Σ*** **X** ) *·* det ( 2 *π* ***Σ*** **Y** ) � 2


1

=
�det ( 2 *π* ***Σ*** **X** )


2 [(] ***[x]*** *[ −]* ***[µ]*** **[X]** [)] *[⊤]* ***[Σ]*** *[−]* **X** [1] [(] ***[x]*** *[ −]* ***[µ]*** **[X]** [)]


*−* [1]

2 [(] ***[y]*** *[ −]* ***[µ]*** **[Y]** [)] *[⊤]* ***[Σ]*** *[−]* **Y** [1] [(] ***[y]*** *[ −]* ***[µ]*** **[Y]** [)] �


= *N* ( ***x*** ; ***µ*** **X**, ***Σ*** **X** ) *· N* ( ***y*** ; ***µ*** **Y**, ***Σ*** **Y** ) .

fundamentals 39


In particular, this implies that random variables that are jointly Gaussian are mutually independent iff the covariance matrix of the corresponding multivariate normal distribution is diagonal.

**Theorem 1.69** (Marginal and conditional distribution) **.** *Consider the*
*Gaussian random vector* **X** *and fix index sets A* *⊆* [ *n* ] *and B* *⊆* [ *n* ] *. Then,*
*we have that for any such* marginal distribution *,*


**X** *A* *∼N* ( ***µ*** *A*, ***Σ*** *AA* ), (1.116) By ***µ*** *A* we denote [ *µ* *i* 1, . . ., *µ* *i* *k* ] where
*A* = *{* *i* 1, . . . *i* *k* *}* . ***Σ*** *AA* is defined
analogously.


(1.116)


*and that for any such* conditional distribution *,*


**X** *A* *|* **X** *B* = ***x*** *B* *∼N* ( ***µ*** *A* *|* *B*, ***Σ*** *A* *|* *B* ) *where* (1.117)

***µ*** *A* *|* *B* = . ***µ*** *A* + ***Σ*** *AB* ***Σ*** *−* *BB* 1 [(] ***[x]*** *[B]* *[ −]* ***[µ]*** *[B]* [)] [,] (1.118) Here, ***µ*** *A* characterizes the prior belief


***Σ*** *A* *|* *B* = . ***Σ*** *AA* *−* ***Σ*** *AB* ***Σ*** *−* *BB* 1 ***[Σ]*** *[BA]* [.] (1.119)

Observe, that the variance can only shrink! Moreover, how much
the variance is reduced depends purely on *where* the observations are
made (e.g., the choice of *B* ) but not on *what* the observations are. Also
observe that ***µ*** *A* *|* *B* depends affinely on ***µ*** *B* .


and ***Σ*** *AB* ***Σ*** *[−]* *BB* [1] [(] ***[x]*** *[B]* *[ −]* ***[µ]*** *[B]* [)] [ represents “how]
different” ***x*** *B* is from what is expected.








book” (Petersen et al.)

40 probabilistic artificial intelligence

**Fact 1.71** (Closedness under affine transformations) **.** *Given* ***A*** *∈* **R** *[m]* *[×]* *[n]*

*and* ***b*** *∈* **R** *[m]* *,*

***A*** **X** + ***b*** *∼N* ( ***Aµ*** + ***b***, ***AΣ*** ***A*** *[⊤]* ) . (1.120)

**Fact 1.72** (Additivity) **.** *Given two independent Gaussian random vectors*
**X** *∼N* ( ***µ***, ***Σ*** ) *and* **X** *[′]* *∼N* ( ***µ*** *[′]*, ***Σ*** *[′]* ) *in* **R** *[n]* *,*

**X** + **X** *[′]* *∼N* ( ***µ*** + ***µ*** *[′]*, ***Σ*** + ***Σ*** *[′]* ) . (1.121)

These properties are unique to Gaussians and a reason for why they
are widely used for learning and inference.

We finish by relating Gaussian random vectors back to our definition
of the multivariate Gaussian distribution (1.114).

**Lemma 1.73.** *Let X* 1, . . ., *X* *n* *∼N* ( 0, 1 ) *be i.i.d. Then the random vector*

[ *X* 1, . . ., *X* *n* ] *is a Gaussian random vector.*

*Proof.* By the closedness under linear transformations (1.120) and the
additivity of Gaussians (1.121), any linear combination of *X* 1, . . ., *X* *n*

follows a Gaussian distribution.

**Lemma 1.74.** *The random vector* **X** *∼N* ( ***µ***, ***Σ*** ) *where* ***Σ*** *is positive definite*
*is equivalently characterized as*

1 / 2
**X** = ***Σ*** **Y** + ***µ*** . (1.122)

*where* **Y** *∼N* ( **0**, ***I*** ) *and* ***Σ*** [1] [/] [2] *is the square root of* ***Σ*** *.* [32] 32 More details on the square root of
a symmetric and positive semi-definite
matrix can be found in section 1.6.3.
*Proof.* By eq. (1.120),

1 / 2 1 / 2 1 / 2 1 / 2 *[⊤]*
**X** = ***Σ*** **Y** + ***µ*** *∼N* ( ***Σ*** **0** + ***µ***, ***Σ*** ***IΣ*** ) = *N* ( ***µ***, ***Σ*** ) .

**Theorem 1.75. X** *∼N* ( ***µ***, ***Σ*** ) *is a Gaussian random vector.*

*Proof.* Equation (1.122) gives an affine transformation of a standard
Gaussian random vector **Y** *∼N* ( **0**, ***I*** ) to **X** .


fundamentals 41


*1.6.2* *Conditional Linear Gaussians*

It is often useful to consider a Gaussian random variable *X* in terms
of another Gaussian random variable *Y* . Suppose *X* *∼N* ( *µ* *X*, *σ* *X* [2] [)] [ and]
*Y* *∼N* ( *µ* *Y*, *σ* *Y* [2] [)] [ are jointly Gaussian. Then, using the closed-form ex-]
pression of Gaussian conditionals (1.117), their conditional distribution
is given as

*X* *|* *Y* = *y* *∼N* ( *µ* *X* *|* *Y*, *σ* *X* [2] *|* *Y* [)] where (1.123)

*µ* *X* *|* *Y* = . *µ* *X* + *σ* *XY* *σ* *Y* *−* 2 [(] *[y]* *[ −]* *[µ]* *[Y]* [)] [,] (1.124)

*σ* *X* [2] *|* *Y* = . *σ* *X* 2 *[−]* *[σ]* *[XY]* *[σ]* *Y* *[−]* [2] *[σ]* *[YX]* [.] (1.125)

Here, *σ* *XY* refers to the covariance of *X* and *Y* . Again, note that *µ* *X* *|* *Y*
depends affinely on *µ* *X* .

Observe that this admits an equivalent characterization of *X* as an
affine function of *Y* with added independent Gaussian noise. Formally,
we define

*X* = . *aY* + *b* + *ϵ* where (1.126)

*a* = . *σ* *XY* *σ* *Y* *−* 2 [,] (1.127)

*b* = . *µ* *X* *−* *σ* *XY* *σ* *Y* *−* 2 *[µ]* *[Y]* [,] (1.128)

*ϵ* *∼N* ( 0, *σ* *X* [2] *|* *Y* [)] [.] (1.129)

It directly follows from the closedness of Gaussians under linear transformations (1.120) that the characterization of *X* via (1.126) is equivalent to *X* *∼N* ( *µ* *X*, *σ* *X* [2] [)] [, and hence,] *[ any]* [ Gaussian] *[ X]* [ can be modeled as]
a *conditional linear Gaussian*, i.e., a linear function of another Gaussian

*Y* with additional independent Gaussian noise.

*1.6.3* *Quadratic Forms*

**Definition 1.77** (Quadratic form) **.** Given a symmetric matrix ***A*** *∈* **R** *[n]* *[×]* *[n]*,
the *quadratic form* induced by ***A*** is defined as


*n*
###### f : R [n] → R, x �→ x [⊤] Ax = ∑

*i* = 1


*n*
###### ∑ A ( i, j ) · x ( i ) · x ( j ) . (1.130)

*j* = 1


We call ***A*** a *positive definite* matrix if all eigenvalues of ***A*** are positive.
Equivalently, we have *f* ( ***x*** ) *>* 0 for all ***x*** *∈* **R** *[n]* *\ {* **0** *}* and *f* ( **0** ) = 0.
Similarly, ***A*** is called *positive semi-definite* if all eigenvalues of ***A*** are
non-negative, or equivalently, if *f* ( ***x*** ) *≥* 0 for all ***x*** *∈* **R** *[n]* . In particular,
if ***A*** is positive definite then � *f* ( ***x*** ) is a norm (called the *Mahalanobis*

*norm* induced by ***A*** ), and is often denoted by *∥* ***x*** *∥* ***A*** .

42 probabilistic artificial intelligence

If ***A*** is positive definite, then the sublevel sets of its induced quadratic
form *f* are convex ellipsoids. Not coincidentally, the same is true for
the sublevel sets of the PDF of a normal distribution *N* ( ***µ***, ***Σ*** ), which
we have seen an example of in fig. 1.12. Hereby, the axes of the ellipsoid and their corresponding squared lengths are the eigenvectors and
eigenvalues of ***Σ***, respectively.



One important property of positive definiteness of ***Σ*** is that ***Σ*** can
be decomposed into the product of a lower-triangular matrix with its
transpose. This is known as the *Cholesky decomposition* .

**Fact 1.79** (Cholesky decomposition, symmetric matrix-form) **.** *For any*
*symmetric and positive (semi-)definite matrix* ***A*** *∈* **R** *[n]* *[×]* *[n]* *, there is a decom-*
*position of the form*

***A*** = *LL* *[⊤]* (1.131)

*where* *L ∈* **R** *[n]* *[×]* *[n]* *is lower triangular and positive (semi-)definite.*

We will not prove this fact, but it is not hard to see that a decomposition exists (it takes more work to show that *L* is lower-triangular).

Let ***A*** be a symmetric and positive (semi-)definite matrix. By the spectral theorem of symmetric matrices, ***A*** = ***VΛV*** *[⊤]*, where ***Λ*** is a diagonal
matrix of eigenvalues and ***V*** is an orthonormal matrix of corresponding eigenvectors. Consider the matrix

***A*** 1 / 2 = . ***VΛ*** 1 / 2 ***V*** *⊤* (1.132)

where ***Λ*** [1] [/] [2] = . diag *{* � ***Λ*** ( *i*, *i* ) *}*, also called the *square root* of ***A*** . Then,

1 / 2 1 / 2 1 / 2 *⊤* 1 / 2 *⊤*
***A*** ***A*** = ***VΛ*** ***V*** ***VΛ*** ***V***

= ***VΛ*** 1 / 2 ***Λ*** 1 / 2 ***V*** *⊤*

= ***VΛV*** *[⊤]* = ***A*** . (1.133)

It is immediately clear from the definition that ***A*** [1] [/] [2] is also symmetric
and positive (semi-)definite.

Quadratic forms of positive semi-definite matrices are a generalization
of the Euclidean norm, as

1 / 2 1 / 2 1 / 2 *⊤* 1 / 2 1 / 2 2
*∥* ***x*** *∥* [2] ***A*** [=] ***[ x]*** *[⊤]* ***[Ax]*** [ =] ***[ x]*** *[⊤]* ***[A]*** ***A*** ***x*** = ( ***A*** ***x*** ) ***A*** ***x*** = *∥* ***A*** ***x*** *∥* 2 [,] (1.134)

fundamentals 43

and in particular,

log *N* ( ***x*** ; ***µ***, ***Σ*** ) = *−∥* ***x*** *−* ***µ*** *∥* ***Λ*** [2] [+] [ const] and (1.135) where ***Λ*** = ***Σ*** *[−]* [1]

log *N* ( ***x*** ; **0**, ***I*** ) = *−∥* ***x*** *∥* [2] 2 [+] [ const.] (1.136)

part I
## *Probabilistic Machine Learning*

### *2* *Bayesian Linear Regression*

*2.1*
*Linear Regression*


We consider the supervised learning setting that we have introduced
in section 1.2. Given a set of ( ***x***, *y* ) pairs, linear regression aims to find
a linear model that fits the data optimally. [1] Given the linear model 1 As we have discussed in section 1.2,
regression models can also be used for

*y* *≈* ***w*** *[⊤]* ***x*** = . *f* ( ***x*** ; ***w*** ) classification. One example of a lin
ear regression model for classification is

for ***x*** *∈* **R** *[d]* and labeled training data *{* ( ***x*** *i*, *y* *i* ) *}* *i* *[n]* = 1 [, we want to find] Bayesian logistic regression, which wewill see in example 5.3.


*y* *≈* ***w*** *[⊤]* ***x*** = . *f* ( ***x*** ; ***w*** )


for ***x*** *∈* **R** *[d]* and labeled training data *{* ( ***x*** *i*, *y* *i* ) *}* *i* *[n]* = 1 [, we want to find]
optimal weights ***w*** .



We define the *design matrix*,

***X*** = .







***x*** 1

...

***x*** *n*


*y*

2

0

*−* 2

*−* 2 0 2

*x*

Figure 2.1: Example of linear regression
with the least squares estimator (shown
in blue).




 *∈* **R** *n* *×* *d*, (2.1)


as the collection of inputs and the vector ***y*** = [ . *y* 1 *· · ·* *y* *n* ] *⊤* *∈* **R** *n* as the
collection of labels. For each noisy observation ( ***x*** *i*, *y* *i* ), we define the
value of the approximation of our model, *f* *i* = . ***w*** *⊤* ***x*** *i* . Our model at the
inputs ***X*** is described by the vector ***f*** = [ . *f* 1 *· · ·* *f* *n* ] *⊤* . It requires little
calculation to verify that ***f*** = ***Xw*** .

48 probabilistic artificial intelligence

There are many ways of estimating ***w*** from data, the most common
being the *least squares estimator*,


***w*** ˆ ls = . arg min
***w*** *∈* **R** *[d]*


*n*
###### ∑ ( y i − w [⊤] x i ) [2] = arg min ∥ y − Xw ∥ [2] 2 [,] (2.2)

*i* = 1 ***w*** *∈* **R** *[d]*


minimizing the squared difference between the labels and predictions
of the model. A slightly different estimator is used for *ridge regression*,

***w*** ˆ ridge = . arg min *∥* ***y*** *−* ***Xw*** *∥* [2] 2 [+] *[ λ]* *[ ∥]* ***[w]*** *[∥]* 2 [2] (2.3)
***w*** *∈* **R** *[d]*


where *λ* *>* 0. The squared *ℓ* 2 regularization term *λ* *∥* ***w*** *∥* [2] 2 [penalizes]
large ***w*** and thus reduces the complexity of the resulting model. This
reduces the potential of overfitting to the training data. [2] 2 Ridge regression is more robust than
standard linear regression in the presence of multicollinearity. *Multicollinear-*

inputs are highly correlated. In this case,

sical linear regression is highly volatile

ization of ridge regression reduces this

weights towards 0.








*2.2* *Uncertainty*

Linear regression might make it seem as though for any data *D*, there
is a clear and unique choice for an optimal regression model. Generally, this is true when our objective is simply to find optimal weights

with respect to some estimator (e.g., the least squares estimator). However, in practice, our data *D* is merely a sample of the process we are
modeling. In these cases, we are looking for models that generalize to

unseen data.

Therefore, it is useful to express the uncertainty about our model due
to the lack of data. This uncertainty is commonly referred to as the
*epistemic uncertainty* . A natural Bayesian approach is to represent epistemic uncertainty with a probability distribution over models. Intuitively, the variance of this distribution measures our uncertainty about
the model and its mode corresponds to our current best (point) estimate. In this and the following chapters, we will explore this approach
extensively.

Usually, there is another source of uncertainty called *aleatoric uncer-*
*tainty*, which originates directly from the process that we are modeling. This uncertainty is the noise in the labels that cannot be explained
by the inputs. Often, this uncertainty is called “irreducible noise” or
simply “(label) noise”.

It is a practical modeling choice how much inaccuracy to attribute
to epistemic or aleatoric uncertainty. Generally, when a poor model
is used to explain a process, more inaccuracy has to be attributed to

irreducible noise.

A very informative interpretation of epistemic and aleatoric uncertainty is in terms of the law of total variance (1.50),


Var [ *y* *[⋆]* *|* ***x*** *[⋆]* ] = **E** ***θ*** �Var *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ] � + Var ***θ*** � **E** *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ] �


. (2.6)


bayesian linear regression 49

*y*

2.5

0.0

*−* 2.5

*−* 5.0

*−* 2 0 2

*x*


Here, the mean variability of predictions *y* *[⋆]* averaged across all models ***θ*** can be interpreted as the aleatoric uncertainty. In contrast, the
variability of the mean prediction *y* *[⋆]* under each model ***θ*** can be interpreted as the epistemic uncertainty.


Figure 2.2: The same linear regression

preted as the epistemic uncertainty. model as in fig. 2.1 assuming *σ* *n* = 1.

The estimated variance of the weights
(i.e., epistemic uncertainty) is shown in

light gray.

and also has to be estimated.


evaluated at ***x*** *[⋆]*

50 probabilistic artificial intelligence








*2.3* *Weight-space View*

There are two views on Bayesian linear regression, the weight-space
view and the function-space view. We begin by discussing the common view in terms of a weight parameterization. In section 2.6, we
introduce the interpretation in terms of functions, which will naturally
lead us to Gaussian processes.

The most immediate and natural Bayesian interpretation of linear regression is to simply impose a prior on the weights ***w*** . We will use a
Gaussian prior,

***w*** *∼N* ( **0**, *σ* *p* [2] ***[I]*** [)] [,] (2.9)







Figure 2.3: Directed graphical model of
Bayesian linear regression in plate notation.


and a Gaussian likelihood, [4] 4 The choice of hyperparameters such as
the prior variance *σ* *p* [2] [and the label noise]
*y* *i* *|* ***x*** *i*, ***w*** *∼N* ( ***w*** *[⊤]* ***x*** *i*, *σ* *n* [2] [)] [.] (2.10) *σ* *n* [2] [is discussed in greater detail in sec-]
tion 4.4.

Note that this likelihood models the noisy observations

*y* *i* = ***w*** *[⊤]* ***x*** *i* + *ϵ* *i* (2.11)

where *ϵ* *i* *∼N* ( 0, *σ* *n* [2] [)] [ is homoscedastic noise.] [5] 5 Homoscedastic noise is noise that is independent of the input ***x*** *i* . *ϵ* *i* is often
called *additive white Gaussian noise* . Refer to section 7.2.1 for a more thorough
discussion of noise.

bayesian linear regression 51







*2.3.1* *Learning*

Next, let us derive the posterior distribution over the weights.

log *p* ( ***w*** *|* ***x*** 1: *n*, *y* 1: *n* )

= log *p* ( ***w*** ) + log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***w*** ) + const by Bayes’ rule (1.58)

*n*
###### = log p ( w ) + ∑ log p ( y i | x i, w ) + const using independence of the samples

*i* = 1

52 probabilistic artificial intelligence


1

= *−* 2 [1] *σ* *p* [2] *∥* ***w*** *∥* [2] 2 *[−]* 2 *σ* *n* [2]


*n*
###### ∑ ( y i − w [⊤] x i ) [2] + const using the Gaussian prior and likelihood

*i* = 1


1

= *−* 2 [1] *σ* *p* [2] *∥* ***w*** *∥* [2] 2 *[−]* 2 *σ* *n* [2] *∥* ***y*** *−* ***Xw*** *∥* [2] 2 + const using ∑ *i* *[n]* = 1 [(] *[y]* *[i]* *[−]* ***[w]*** *[⊤]* ***[x]*** *[i]* [)] [2] [ =] *[ ∥]* ***[y]*** *[ −]* ***[Xw]*** *[∥]* [2] 2


1

= *−* [1] ***w*** *[⊤]* ***w*** *−*

2 *σ* *p* [2] 2 *σ* *n* [2]


***w*** *[⊤]* ***X*** *[⊤]* ***Xw*** *−* 2 ***y*** *[⊤]* ***Xw*** + ***y*** *[⊤]* ***y*** + const.
� �


By completing the square, we obtain [6] 6 For more details, refer to theorem 9.1
of “Mathematics for machine learning”
(Deisenroth et al.).
= *−* [1] (2.13)

2 [(] ***[w]*** *[ −]* ***[µ]*** [)] *[⊤]* ***[Σ]*** *[−]* [1] [(] ***[w]*** *[ −]* ***[µ]*** [) +] [ const]

where

***µ*** = . � ***X*** *[⊤]* ***X*** + *σ* *n* [2] *[σ]* *p* *[−]* [2] ***[I]*** � *−* 1 ***X*** *[⊤]* ***y*** = *σ* *n* *[−]* [2] ***[Σ][X]*** *[⊤]* ***[y]*** [,] (2.14)

***Σ*** = . � *σ* *n* *[−]* [2] ***[X]*** *[⊤]* ***[X]*** [ +] *[ σ]* *p* *[−]* [2] ***[I]*** � *−* 1 . (2.15)

As the above is a quadratic form in ***w***, it follows that the posterior
distribution is a Gaussian (cf. remark 1.78),

***w*** *|* ***x*** 1: *n*, *y* 1: *n* *∼N* ( ***µ***, ***Σ*** ) . (2.16)

This also shows that Gaussians with known variance and linear like
lihood are self-conjugate, a property that we had hinted at in section 1.3.2. It can be shown more generally that Gaussians with known
variance are self-conjugate to any Gaussian likelihood. [7] 7 Kevin P Murphy. Conjugate bayesian
analysis of the gaussian distribution. *def*,
For general distributions the posterior will not be closed-form. This is 1(2 *σ* 2):16, 2007
a very special property of Gaussians!



bayesian linear regression 53











Note that using point estimates like the MAP estimate does not quantify uncertainty in the weights. The MAP estimate simply collapses all
mass of the posterior around its mode. This is especially harmful when
we are unsure about the best model (i.e., the epistemic uncertainty is
large).

*2.3.2* *Inference*

To quantify uncertainty, we can use *Bayesian linear regression* . For
a Gaussian prior and likelihood, we again have closed-form formulas. For a test point ***x*** *[⋆]*, we define the (model-)predicted point *f* *[⋆]* = .

54 probabilistic artificial intelligence

***w*** *[⊤]* ***x*** *[⋆]* . Using the closedness of Gaussians under linear transformations (1.120),

*f* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* *∼N* ( ***µ*** *[⊤]* ***x*** *[⋆]*, ***x*** *[⋆][⊤]* ***Σx*** *[⋆]* ) . (2.21)

Note that this does not take into account the noise in the labels *σ* *n* [2] [.]
Hence, for the label prediction *y* *[⋆]*, we obtain

*y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* *∼N* ( ***µ*** *[⊤]* ***x*** *[⋆]*, ***x*** *[⋆][⊤]* ***Σx*** *[⋆]* + *σ* *n* [2] [)] [.] (2.22)

This already highlights a decomposition of uncertainty wherein ***x*** *[⋆][⊤]* ***Σx*** *[⋆]*

corresponds to the epistemic uncertainty in the model and *σ* *n* [2] [corre-]
sponds to the aleatoric uncertainty (i.e., label noise).












*−* 0.5 0.0 0.5 1.0 1.5

*x*


bayesian linear regression 55

Figure 2.4: Comparison of **linear regres-**
*y*
**sion (MLE)**, **ridge regression (MAP es-**
**timate)**, and **Bayesian linear regression**
when the data is generated according to

*y* *|* ***w***, ***x*** *∼N* ( ***w*** *[⊤]* ***x***, *σ* *n* [2] [)] [.]


3

2

1

0

*−* 1


*y*

*−* 0.5 0.0 0.5 1.0 1.5

*x*


6

4

2

0

*−* 2

*−* 4


The **true mean** is shown in blue, the
MLE in green, and the MAP estimate in
red. The dark gray area denotes the epistemic uncertainty of Bayesian linear regression and the light gray area the additional homoscedastic noise (i.e., aleatoric
uncertainty). Plausible realizations are
shown in black. On the left, *σ* *n* = 0.15.
On the right, *σ* *n* = 0.7.




*2.4* *Recursive Bayesian Updates*

A more general feature of supervised Bayesian learning, which often
leads to efficient algorithms, is its recursive structure. Suppose we are
given a prior *p* ( ***θ*** ) and observations *y* 1: *n* . We define

*p* [(] *[t]* [)] ( ***θ*** ) = . *p* ( ***θ*** *|* *y* 1: *t* ) (2.25)

to be the posterior after the first *t* observations. We therefore have
*p* [(] [0] [)] ( ***θ*** ) = *p* ( ***θ*** ) . Now, suppose that we have already computed *p* [(] *[t]* [)] ( ***θ*** )
and observe *y* *t* + 1 . We can then recursively update the posterior as
follows,

*p* [(] *[t]* [+] [1] [)] ( ***θ*** ) = *p* ( ***θ*** *|* *y* 1: *t* + 1 )

= [1] using Bayes’ rule (1.58)

*Z* *[p]* [(] ***[θ]*** *[ |]* *[ y]* [1:] *[t]* [)] *[p]* [(] *[y]* *[t]* [+] [1] *[ |]* ***[ θ]*** [,] *[ y]* [1:] *[t]* [)]

= [1] (2.26) using *y* *t* + 1 *⊥* *y* 1: *t* *|* ***θ***, see fig. 2.3

*Z* *[p]* [(] *[t]* [)] [(] ***[θ]*** [)] *[p]* [(] *[y]* *[t]* [+] [1] *[ |]* ***[ θ]*** [)] [.]

Intuitively, the posterior distribution “absorbs”, or “summarizes” all

seen data.

Thus, as data arrives online (i.e., in “real-time”), we can obtain the
new posterior and use it to replace our prior. In general, the posterior

56 probabilistic artificial intelligence

distribution can be complicated. For Bayesian linear regression with
a Gaussian prior and likelihood, however, the posterior distribution is
also a Gaussian,

*p* [(] *[t]* [)] ( ***w*** ) = *N* ( ***w*** ; ***µ*** [(] *[t]* [)], ***Σ*** [(] *[t]* [)] ), (2.27)

which can be stored efficiently using only *O* � *d* [2] [�] parameters.



This interpretation of Bayesian linear regression as an online algorithm
also highlights similarities to other sequential models such as Kalman
filters, which we discuss in chapter 3. In example 3.2, we show that
online Bayesian linear regression is, in fact, an example of a Kalman
filter.

*2.5* *Non-linear Regression*

We can use linear regression not only to learn linear functions. The
trick is to apply a non-linear transformation ***ϕ*** : **R** *[d]* *→* **R** *[e]* to the features ***x*** *i*, where *d* is the dimension of the input space and *e* is the
dimension of the designed *feature space* . We denote the design matrix comprised of transformed features by ***Φ*** *∈* **R** *[n]* *[×]* *[e]* . Note that if the
feature transformation ***ϕ*** is the identity function then ***Φ*** = ***X*** .


*y*

10

5

0

*−* 5

*−* 2 0 2

*x*

Figure 2.5: Applying linear regression
with a feature space of polynomials of
degree 10. The **least squares estimate** is
shown in blue, **ridge regression** in red,
and **lasso** in green.

bayesian linear regression 57




number of ways to choose *i* times from

sider the following encoding: We take
a sequence of *d* + *i* *−* 1 spots. Select
The example of polynomials highlights that it may be inefficient to ing any subset of *i* spots, we interpret

the remaining *d* *−* 1 spots as “barriers”

keep track of the weights ***w*** *∈* **R** *[e]* when *e* is large, and that it may be separating each of the *d* items. The seuseful to instead consider a reparameterization which is of dimension lected spots correspond to the number
rather than of the feature dimension. of times each item has been selected. For

example, if 2 items are to be selected out
of a total of 4 items with replacement,
one possible configuration is “ *◦|| ◦|* ”

*6* *Function-space View* where *◦* denotes a selected spot and *|*

denotes a barrier. This configuration en
Let us look at Bayesian linear regression through a different lens. Pre- codes that the first and third item have

each been chosen once. The number of

viously, we have been interpreting it as a distribution over the weights possible configurations — each encoding
of a linear function ***f*** = ***Φw*** . The key idea is that for a finite set a unique outcome — is therefore ( *[d]* [+] *i* *[i]* *[−]* [1] ) .



The example of polynomials highlights that it may be inefficient to
keep track of the weights ***w*** *∈* **R** *[e]* when *e* is large, and that it may be
useful to instead consider a reparameterization which is of dimension

*n* rather than of the feature dimension.


*2.6* *Function-space View*


Let us look at Bayesian linear regression through a different lens. Previously, we have been interpreting it as a distribution over the weights
***w*** of a linear function ***f*** = ***Φw*** . The key idea is that for a finite set
of inputs (ensuring that the design matrix is well-defined), we can
equivalently consider a distribution directly over the estimated function values ***f*** .


*y*


|Col1|fn f1|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||
|||||||||||||
|||||||||||||
|||||||||||||


*x* 1 *x* *n*

*x*


You may say, that nothing has changed — and you would be right:
that is precisely the point. However, instead of considering a prior
over the weights ***w*** *∼N* ( **0**, *σ* *p* [2] ***[I]*** [)] [ as we have done previously, we now]
impose a prior directly on the values of our model at the observations.
Using that Gaussians are closed under linear maps (1.120), we obtain
the equivalent prior


2

0

*−* 2

*−* 4


***f*** *|* ***X*** *∼N* ( ***Φ*** **E** [ ***w*** ], ***Φ*** Var [ ***w*** ] ***Φ*** *[⊤]* ) = *N* ( **0**, *σ* *p* [2] ***[ΦΦ]*** *[⊤]*
����
***K***


) (2.29)


Figure 2.6: An illustration of the
function-space view. The model is described by the points ( *x* *i*, *f* *i* ) .

58 probabilistic artificial intelligence

where ***K*** *∈* **R** *[n]* *[×]* *[n]* is the so-called *kernel matrix* . Observe that the entries
of the kernel matrix can be expressed as ***K*** ( *i*, *j* ) = *σ* *p* [2] *[·]* ***[ ϕ]*** [(] ***[x]*** *i* [)] *[⊤]* ***[ϕ]*** [(] ***[x]*** *j* [)] [.]

The kernel matrix ***K*** has entries only for the finite set of observed
inputs. However, in principle, we could have observed any input, and
this motivates the definition of the *kernel function*

*k* ( ***x***, ***x*** *[′]* ) = . *σ* *p* 2 *[·]* ***[ ϕ]*** [(] ***[x]*** [)] *[⊤]* ***[ϕ]*** [(] ***[x]*** *[′]* [)] (2.30)

for arbitrary inputs ***x*** and ***x*** *[′]* . A kernel matrix is simply a finite “view”
of the kernel function,




(2.31)



***K*** =







*k* ( ***x*** 1, ***x*** 1 ) *· · ·* *k* ( ***x*** 1, ***x*** *n* )

... ... ...

*k* ( ***x*** *n*, ***x*** 1 ) *· · ·* *k* ( ***x*** *n*, ***x*** *n* )


Observe that by definition of the kernel matrix in eq. (2.29), the kernel

matrix is a covariance matrix and the kernel function measures the

covariance of the function values *f* ( ***x*** ) and *f* ( ***x*** *[′]* ) given inputs ***x*** and ***x*** *[′]* .
That is,

*k* ( ***x***, ***x*** *[′]* ) = Cov� *f* ( ***x*** ), *f* ( ***x*** *[′]* ) �. (2.32)

Moreover, note that we have reformulated [9] the learning algorithm 9 we often say “kernelized”
such that the feature space is now *implicit* in the choice of kernel, and
the kernel is defined by inner products of (nonlinearly transformed)
inputs. In other words, the choice of kernel implicitly determines the
class of functions that ***f*** is sampled from (without expressing the functions explicitly in closed-form), and encodes our prior beliefs. This is

known as the *kernel trick* .

*2.6.1* *Learning and Inference*

We have already kernelized the Bayesian linear regression prior. The
posterior distribution ***f*** *|* ***X***, ***y*** is again Gaussian due to the closedness
properties of Gaussians, analogously to our derivation of the prior
kernel matrix in eq. (2.29).

It remains to show that we can also rely on the kernel trick during
inference. Given the test point ***x*** *[⋆]*, we define


�


�


�


***f***
� *f* *[⋆]*


***Φ*** ˜ = .


***Φ***
� ***ϕ*** ( ***x*** *[⋆]* ) *[⊤]*


, ***y*** ˜ = .


.


***y***
� *y* *[⋆]*


, ***f*** ˜ = .


We immediately obtain ***f*** [˜] = ***Φ*** [˜] ***w*** . Analogously to our analysis of inference from the weight-space view, we add the label noise to obtain
. *⋆* *⊤* 2
the estimate ˜ ***y*** = ***f*** [˜] + ˜ ***ϵ*** where ˜ ***ϵ*** = [ *ϵ* 1 *· · ·* *ϵ* *n* *ϵ* ] *∼N* ( **0**, *σ* *n* ***[I]*** [)] [ is the]

bayesian linear regression 59


independent label noise. Applying the same reasoning as we did for
the prior, we obtain

***f*** ˜ *|* ***X***, ***x*** *[⋆]* *∼N* ( **0**, ˜ ***K*** ) (2.33)

where ***K*** [˜] = . *σ* *p* 2 ***[Φ]*** [˜] [ ˜] ***[Φ]*** *[⊤]* [. Adding the label noise yields]

***y*** ˜ *|* ***X***, ***x*** *[⋆]* *∼N* ( **0**, ***K*** [˜] + *σ* *n* [2] ***[I]*** [)] [.] (2.34)

Finally, we can conclude from the closedness of Gaussian random vectors under conditional distributions (1.117) that the predictive posterior *y* *[⋆]* *|* ***x*** *[⋆]*, ***X***, ***y*** follows again a normal distribution. We will do a full
derivation of the posterior and predictive posterior in section 4.1.

*2.6.2* *Efficient Polynomial Regression*

But how does the kernel trick address our concerns about efficiency
raised in section 2.5? After all, computing the kernel for a feature
space of dimension *e* still requires computing sums of length *e* which
is prohibitive when *e* is large. The kernel trick opens up a couple of

new doors for us:

1. For certain feature transformations ***ϕ***, we may be able to find an
easier to compute expression equivalent to ***ϕ*** ( ***x*** ) *[⊤]* ***ϕ*** ( ***x*** *[′]* ) .
2. If this is not possible, we could approximate the inner product by
an easier to compute expression.
3. Or, alternatively, we may decide not to care very much about the
exact feature transformation and simply experiment with kernels
that induce *some* feature space (which may even be infinitely dimensional).

We will explore the third approach when we revisit kernels in section 4.3. A polynomial feature transformation can be computed efficiently in closed-form.

**Fact 2.14.** *For the polynomial feature transformation* ***ϕ*** *up to degree m from*
*example 2.13, it can be shown that*

***ϕ*** ( ***x*** ) *[⊤]* ***ϕ*** ( ***x*** *[′]* ) = ( 1 + ***x*** *[⊤]* ***x*** *[′]* ) *[m]* (2.35)

*up to constant factors.*

For example, for input dimension 2, the kernel ( 1 + ***x*** *[⊤]* ***x*** *[′]* ) [2] corresponds to the feature vector ***ϕ*** ( ***x*** ) = [ 1 *√* 2 *x* 1 *√* 2 *x* 2 *√* 2 *x* 1 *x* 2 *x* 1 [2] *[x]* 2 [2] []] *[⊤]* [.]


2 *x* 1 *√*


2 *x* 2 *√*


2 *x* 1 *x* 2 *x* 1 [2] *[x]* 2 [2] []] *[⊤]* [.]

### *3* *Kalman Filters*

Before we continue in chapter 4 with the function-space view of regression, we want to look at a seemingly different but very related
problem. Kalman filters are an example of Bayesian learning and inference in the *state space model*, where we want to keep track of the
state of an agent over time based on noisy observations. In this model,
we have a sequence of (hidden) states ( **X** *t* ) *t* *∈* **N** 0 where **X** *t* is in **R** *[d]* and
a sequence of observations ( **Y** *t* ) *t* *∈* **N** 0 where **Y** *t* is in **R** *[m]* .

The process of keeping track of the state using noisy observations is
also known as *Bayesian filtering* or *recursive Bayesian estimation* . We will
discuss this process more broadly in the next section. A Kalman filter
is simply a Bayes filter using a Gaussian distribution over the states

and conditional linear Gaussians to describe the evolution of states

and observations.

**Definition 3.1** (Kalman filter) **.** A *Kalman filter* is specified by a Gaussian prior over the states,

**X** 0 *∼N* ( ***µ***, ***Σ*** ), (3.1)

and a conditional linear Gaussian *motion model* and *sensor model*,

**X** *t* + 1 = . ***F*** **X** *t* + ***ϵ*** *t* ***F*** *∈* **R** *[d]* *[×]* *[d]*, ***ϵ*** *t* *∼N* ( **0**, ***Σ*** *x* ), (3.2)

**Y** *t* = . ***H*** **X** *t* + ***η*** *t* ***H*** *∈* **R** *[m]* *[×]* *[d]*, ***η*** *t* *∼N* ( **0**, ***Σ*** *y* ), (3.3)

respectively. The motion model is sometimes also called *transition*
*model* or *dynamics model* .

Crucially, Kalman filters assume that ***F*** and ***H*** are known. In general,
***F*** and ***H*** may depend on *t* . Also, ***ϵ*** and ***η*** may have a non-zero mean,
commonly called a “drift”.

Because Kalman filters use conditional linear Gaussians, which we
have already seen in section 1.6.2, their joint distribution (over all variables) is also Gaussian. This means that predicting the future states


Figure 3.1: Directed graphical model of a
Kalman filter with hidden states **X** *t* and
observables **Y** *t* .

62 probabilistic artificial intelligence

of a Kalman filter is simply inference with multivariate Gaussians. In
Bayesian filtering, however, we do not only want to make predictions
occasionally. In Bayesian filtering, we want to *keep track* of states, that
is, predict the current state of an agent online. [1] To do this efficiently, 1 Here, *online* is common terminology to

we need to update our *belief* about the state of the agent recursively, say that we want to perform inference

at time *t* without exposure to times *t* +

similarly to our recursive Bayesian updates in Bayesian linear regres- 1, *t* + 2, . . ., so in “real-time”.
sion (see section 2.4).

From the directed graphical model of a Kalman filter shown in fig. 3.1,
we can immediately gather the following conditional independence
relations, [2] 2 Alternatively, they follow from the definition of the motion and sensor models
**X** *t* + 1 *⊥* **X** 1: *t* *−* 1, **Y** 1: *t* *−* 1 *|* **X** *t*, (3.4) as linear updates.

**Y** *t* *⊥* **X** 1: *t* *−* 1 *|* **X** *t* (3.5)

**Y** *t* *⊥* **Y** 1: *t* *−* 1 *|* **X** *t* *−* 1 . (3.6)

The first conditional independence property is also known as the *Markov*
*property*, which we will return to later in our discussion of Markov
chains and Markov decision processes. This characterization of the
Kalman filter, yields the following factorization of the joint distribu
tion,

*t*
###### p ( x 1: t, y 1: t ) = ∏ p ( x i | x 1: i − 1 ) p ( y i | x 1: t, y 1: i − 1 ) using the product rule (1.14)

*i* = 1

*t*
###### = p ( x 1 ) p ( y 1 | x 1 ) ∏ p ( x i | x i − 1 ) p ( y i | x i ) . (3.7) using the conditional independence

*i* = 2 properties from (3.4), (3.5), and (3.6)


*3.1* *Bayesian Filtering*

As we have already mentioned, *Bayesian filtering* is the process of keeping track of an agent’s state using noisy observations. Bayesian filtering is described by the following recursive scheme with the two
phases, *conditioning* (also called *update* ) and *prediction* .

*p* ( ***x*** 0 ) *p* ( ***x*** *t* + 1 *|* ***y*** 1: *t* )

**prior** **prediction**


kalman filters 63

Figure 3.2: Schematic view of Bayesian
filtering.


**(based on motion model)**


*t* *←* *t* + 1


*p* ( ***x*** *t* *|* ***y*** 1: *t* ) ***y*** *t*

**update** **observation**

**(based on sensor model)**

**Al** **g** **orithm 3.3:** Ba y esian filterin g

**1** start with a prior over initial states *p* ( ***x*** 0 )

**2** `for` *t* = 1 `to` ∞ `do`

**3** assume we have *p* ( ***x*** *t* *|* ***y*** 1: *t* *−* 1 )

**4** compute *p* ( ***x*** *t* *|* ***y*** 1: *t* ) using the new observation ***y*** *t* (conditioning)

**5** compute *p* ( ***x*** *t* + 1 *|* ***y*** 1: *t* ) (prediction)

Let us consider the conditioning step first:

*p* ( ***x*** *t* *|* ***y*** 1: *t* ) = [1] using Bayes’ rule (1.58)

*Z* *[p]* [(] ***[x]*** *[t]* *[ |]* ***[ y]*** [1:] *[t]* *[−]* [1] [)] *[p]* [(] ***[y]*** *[t]* *[ |]* ***[ x]*** *[t]* [,] ***[ y]*** [1:] *[t]* *[−]* [1] [)]

= [1] (3.10) using the conditional independence

*Z* *[p]* [(] ***[x]*** *[t]* *[ |]* ***[ y]*** [1:] *[t]* *[−]* [1] [)] *[p]* [(] ***[y]*** *[t]* *[ |]* ***[ x]*** *[t]* [)] [.]
structure (3.6)

64 probabilistic artificial intelligence

For the prediction step, we obtain,

*p* ( ***x*** *t* + 1 *|* ***y*** 1: *t* ) = *p* ( ***x*** *t* + 1, ***x*** *t* *|* ***y*** 1: *t* ) *d* ***x*** *t* using the sum rule (1.10)
�

= *p* ( ***x*** *t* + 1 *|* ***x*** *t*, *y* 1: *t* ) *p* ( ***x*** *t* *|* *y* 1: *t* ) *d* ***x*** *t* using the product rule (1.14)
�

= *p* ( ***x*** *t* + 1 *|* ***x*** *t* ) *p* ( ***x*** *t* *|* *y* 1: *t* ) *d* ***x*** *t* . (3.11) using the conditional independence
� structure (3.4)

In general, these distributions can be very complicated, but for Gaussians (i.e., Kalman filters) they can be expressed in closed-form.



kalman filters 65



*3.2* *Kalman Update*

Before introducing the general Kalman update, let us consider a simple
example.



drift of 0.



1 2 3 4 5 6

*t*

Figure 3.3: Hidden states during a random walk in one dimension.

66 probabilistic artificial intelligence




The general formulas for the *Kalman update* follow the same logic.
Given the prior belief **X** *t* *|* ***y*** 1: *t* *∼N* ( ***µ*** *t*, ***Σ*** *t* ), we have

**X** *t* + 1 *|* ***y*** 1: *t* + 1 *∼N* ( ***µ*** *t* + 1, ***Σ*** *t* + 1 ) where (3.20)

***µ*** *t* + 1 = . ***Fµ*** *t* + ***K*** *t* + 1 ( ***y*** *t* + 1 *−* ***HFµ*** *t* ), (3.21)

. *⊤*
***Σ*** *t* + 1 = ( ***I*** *−* ***K*** *t* + 1 ***H*** )( ***FΣ*** *t* ***F*** + ***Σ*** *x* ) . (3.22)

Hereby, ***K*** *t* + 1 is the *Kalman gain*,

***K*** *t* + 1 = ( . ***FΣ*** *t* ***F*** *⊤* + ***Σ*** *x* ) ***H*** *⊤* ( ***H*** ( ***FΣ*** *t* ***F*** *⊤* + ***Σ*** *x* ) ***H*** *⊤* + ***Σ*** *y* ) *−* 1 .
(3.23)

Note that ***Σ*** *t* and ***K*** *t* can be computed offline as they are independent
of the observation ***y*** *t* + 1 . ***Fµ*** *t* represents the expected state at time *t* + 1,
and hence, ***HFµ*** *t* corresponds to the expected observation. Therefore,
the term ***y*** *t* + 1 *−* ***HFµ*** *t* measures the error in the predicted observation
and the Kalman gain ***K*** *t* + 1 appears as a measure of relevance of the
new observation compared to the prediction.

*3.3* *Predicting*

Using now that the marginal posterior of **X** *t* is a Gaussian due to the
closedness properties of Gaussians, we have

**X** *t* + 1 *|* ***y*** 1: *t* *∼N* ( ˆ ***µ*** *t* + 1, ***Σ*** [ˆ] *t* + 1 ), (3.24)

and it suffices to compute the prediction mean ˆ ***µ*** *t* + 1 and covariance
matrix ***Σ*** [ˆ] *t* + 1 .

kalman filters 67

For the mean,

***µ*** ˆ *t* + 1 = **E** [ ***x*** *t* + 1 *|* ***y*** 1: *t* ]

= **E** [ ***Fx*** *t* + ***ϵ*** *t* *|* ***y*** 1: *t* ] using the motion model (3.2)

= ***F*** **E** [ ***x*** *t* *|* ***y*** 1: *t* ] using linearity of expectation (1.24) and
**E** [ ***ϵ*** *t* ] = **0**
= ***Fµ*** *t* . (3.25) using the mean of the Kalman update

For the covariance matrix,

***Σ*** ˆ *t* + 1 = **E** ( ***x*** *t* + 1 *−* ***µ*** ˆ *t* + 1 )( ***x*** *t* + 1 *−* ***µ*** ˆ *t* + 1 ) *[⊤]* [��] ***y*** 1: *t* using the definition of the covariance
� � �
matrix (1.46)
= ***F*** **E** ( ***x*** *t* *−* ***µ*** *t* )( ***x*** *t* *−* ***µ*** *t* ) *[⊤]* [��] ***y*** 1: *t* ***F*** *[⊤]* + **E** ***ϵ*** *t* ***ϵ*** *t* *[⊤]* using (3.25), the motion model (3.2) and
� � � � �
that ***ϵ*** *t* is independent of the
= ***FΣ*** *t* ***F*** *[⊤]* + ***Σ*** . (3.26) observations


### *4* *Gaussian Processes*


We obtain Gaussian processes by extending the function-space view of
Bayesian linear regression which we introduced in section 2.6 to infinite domains. [1] We are still concerned with the problem of estimating 1 In section 2.6, our domain was given by

the value of a function *f* : *X →* **R** at arbitrary points ***x*** *[⋆]* *∈X* given the set of weights (weight-space view) or

the set of (noisy) observations (function
training data *{* ***x*** *i*, *y* *i* *}* *i* *[n]* = 1 [, where the labels are assumed to be corrupted] space view).
by homoscedastic Gaussian noise,

*y* *i* = *f* ( ***x*** *i* ) + *ϵ* *i*, *ϵ* *i* *∼N* ( 0, *σ* *n* [2] [)] [.]

As in chapter 2 on Bayesian linear regression, we denote by ***X*** the
design matrix (collection of training inputs) and by ***y*** the vector of
training labels.


**Definition 4.1** (Gaussian process, GP) **.** A *Gaussian process* is an infinite
set of random variables such that any finite number of them are jointly

Gaussian.

We use a set *X* to index the collection of random variables. A Gaussian

process is characterized by a *mean function µ* : *X →* **R** and a *covariance*
*function* (or *kernel function* ) *k* : *X × X →* **R** such that for any *A* = .
*{* ***x*** 1, . . ., ***x*** *m* *} ⊆X*, we have

***f*** *A* = [ . *f* ***x*** 1 *· · ·* *f* ***x*** *m* ] *⊤* *∼N* ( ***µ*** *A*, ***K*** *AA* ) (4.1)

where


*x* *p* ( *f* ( *x* ))

Figure 4.1: A Gaussian process can be
interpreted as an infinite-dimensional
Gaussian over functions. At any location *x* in the domain, this yields a distribution over values *f* ( *x* ) shown in red.
The blue line corresponds to the MAP
estimate (i.e., mean function of the Gaussian process), the dark gray region corresponds to the epistemic uncertainty and
the light gray region denotes the additional aleatoric uncertainty.


, ***K*** *AA* = .









*k* ( ***x*** 1, ***x*** 1 ) *· · ·* *k* ( ***x*** 1, ***x*** *m* )

... ... ...

*k* ( ***x*** *m*, ***x*** 1 ) *· · ·* *k* ( ***x*** *m*, ***x*** *m* )




. (4.2)



***µ*** *A* = .







*µ* ( ***x*** 1 )

...

*µ* ( ***x*** *m* )


We write *f* *∼GP* ( *µ*, *k* ) . In particular, given a mean function, covariance function, and using the homoscedastic noise assumption,

*y* *[⋆]* *|* ***x*** *[⋆]*, *µ*, *k* *∼N* ( *µ* ( ***x*** *[⋆]* ), *k* ( ***x*** *[⋆]*, ***x*** *[⋆]* ) + *σ* *n* [2] [)] [.] (4.3)

70 probabilistic artificial intelligence

The random variable *f* ***x*** represents the value of the function *f* ( ***x*** ) at
location ***x*** *∈X*, we thus write *f* ( ***x*** ) = . *f* ***x*** . Intuitively, a Gaussian
process can be interpreted as a normal distribution over functions —
and is therefore often called an infinite-dimensional Gaussian.

Commonly, for notational simplicity, the mean function is taken to
be zero. Note that for a fixed mean this is not a restriction, as we
can simply apply the zero-mean Gaussian process to the difference
between the mean and the observations. [2] 2 For alternative ways of representing a
mean function, refer to section 2.7 of
“Gaussian processes for machine learn


*4.1* *Learning and Inference*


First, let us look at learning and inference in the context of Gaussian processes. Given a prior *f* *∼GP* ( *µ*, *k* ) and (noisy) observations
*y* *i* = *f* ( ***x*** *i* ) + *ϵ* *i* with *ϵ* *i* *∼N* ( 0, *σ* *n* [2] [)] [ at locations] *[ A]* = . *{* ***x*** 1, . . ., ***x*** *n* *}*, we
can write the joint distribution of the observations *y* 1: *n* and the noisefree prediction *f* *[⋆]* at a test point ***x*** *[⋆]* as

***y***

*|* ***x*** *[⋆]*, ***x*** 1: *n* *∼N* ( ˜ ***µ***, ***K*** [˜] ), where (4.4)

� *f* *[⋆]* �


�


*|* ***x*** *[⋆]*, ***x*** 1: *n* *∼N* ( ˜ ***µ***, ***K*** [˜] ), where (4.4)




.






�


�


***K*** *AA* + *σ* *n* [2] ***[I]*** ***k*** ***x*** *⋆*, *A*
� ***k*** *[⊤]* ***x*** *[⋆]*, *A* *k* ( ***x*** *[⋆]*, ***x*** *[⋆]* )


, ***k*** ***x***, *A* = .







*k* ( ***x***, ***x*** 1 )

...

*k* ( ***x***, ***x*** *n* )


***µ*** ˜ = .


***µ*** *A*
� *µ* ( ***x*** *[⋆]* )


, ***K*** ˜ = .


(4.5)

Deriving the conditional distribution using (1.117), we obtain

*f* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* *∼N* ( *µ* *[⋆]*, *k* *[⋆]* ), where (4.6)

*µ* *[⋆]* = . *µ* ( ***x*** *⋆* ) + ***k*** *⊤* ***x*** *[⋆]*, *A* [(] ***[K]*** *[AA]* [+] *[ σ]* *n* [2] ***[I]*** [)] *[−]* [1] [(] ***[y]*** *A* *[−]* ***[µ]*** *A* [)] [,] (4.7)

*k* *[⋆]* = . *k* ( ***x*** *⋆*, ***x*** *⋆* ) *−* ***k*** *⊤* ***x*** *[⋆]*, *A* [(] ***[K]*** *[AA]* [+] *[ σ]* *n* [2] ***[I]*** [)] *[−]* [1] ***[k]*** ***x*** *[⋆]*, *A* [.] (4.8)

Thus, the Gaussian process posterior is given by

*f* *|* ***x*** 1: *n*, *y* 1: *n* *∼GP* ( *µ* *[′]*, *k* *[′]* ), where (4.9)

*µ* *[′]* ( ***x*** ) = . *µ* ( ***x*** ) + ***k*** *⊤* ***x***, *A* [(] ***[K]*** *[AA]* [+] *[ σ]* *n* [2] ***[I]*** [)] *[−]* [1] [(] ***[y]*** *A* *[−]* ***[µ]*** *A* [)] [,] (4.10)

*k* *[′]* ( ***x***, ***x*** *[′]* ) = . *k* ( ***x***, ***x*** *′* ) *−* ***k*** *⊤* ***x***, *A* [(] ***[K]*** *[AA]* [+] *[ σ]* *n* [2] ***[I]*** [)] *[−]* [1] ***[k]*** ***x*** *[′]*, *A* [.] (4.11)

Observe that analogously to Bayesian linear regression, the posterior
covariance can only decrease when conditioning on additional data,
and is independent of the observations *y* *i* .

gaussian processes 71

We already studied inference in the function-space in section 2.6.1 in
the context of Bayesian linear regression, but did not make the predictive posterior explicit. Using eq. (4.6) and our assumption of homoscedastic noise, the predictive posterior at the test point ***x*** *[⋆]* is

*y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* *∼N* ( *µ* *[⋆]*, *k* *[⋆]* + *σ* *n* [2] [)] [.] (4.12)

*4.2* *Sampling*

Often, we are not interested in the full predictive posterior distribution,
but merely want to obtain samples of our Gaussian process model. We
will briefly examine two approaches.

1. For the first approach, consider a discretized subset of points

.
***f*** = [ *f* 1, . . ., *f* *n* ]

that we want to sample. [3] Note that ***f*** *∼N* ( **0**, ***K*** ) . We have already 3 For example, if we want to render the
seen in eq. (1.122) that function, the length of this vector could
be guided by the screen resolution.

1 / 2
***f*** = ***K*** ***ϵ*** (4.13)

where ***K*** [1] [/] [2] is the square root of ***K*** and ***ϵ*** *∼N* ( **0**, ***I*** ) is standard

Gaussian noise.

2. For the second approach, recall the product rule (1.14),

*n*
###### p ( f 1, . . ., f n ) = ∏ p ( f i | f 1: i − 1 ) .

*i* = 1

That is the joint distribution factorizes neatly into a product where
each factor only depends on the “outcomes” of preceding factors.
We can therefore obtain samples one-by-one, each time conditioning on one more observation:

*f* 1 *∼* *p* ( *f* 1 )

*f* 2 *∼* *p* ( *f* 2 *|* *f* 1 )


*f* 3 *∼* *p* ( *f* 3 *|* *f* 1, *f* 2 )

...

This general approach is known as *forward sampling* .

We will discuss approximate sampling methods in section 4.5.

*4.3* *Kernel Functions*


(4.14)


We have previously introduced kernels in section 2.6 in the context of
a function-space view of Bayesian linear regression. There we have
already seen that kernel functions define the “shape” of the functions

72 probabilistic artificial intelligence

that are realized from a Gaussian distribution over functions, namely a
Gaussian process. In this section, we generalize the notion of kernels to
encompass function classes beyond linear and polynomial functions.

**Definition 4.2** (Kernel function) **.** A *kernel function k* is defined as

*k* ( ***x***, ***x*** *[′]* ) = . Cov� *f* ( ***x*** ), *f* ( ***x*** *[′]* ) � (4.15)

for locations ***x*** and ***x*** *[′]* such that

- *k* ( ***x***, ***x*** *[′]* ) = *k* ( ***x*** *[′]*, ***x*** ) for any ***x***, ***x*** *[′]* *∈X* (symmetry), and

- ***K*** *AA* is positive semi-definite for any *A* *⊆X* .

We say that a kernel function is *positive definite* if ***K*** *AA* is positive definite for any *A* *⊆X* .


The two defining conditions ensure that for any *A* *⊆X*, ***K*** *AA* is
a valid covariance matrix. Indeed, it can be shown that a function

*k* : *X × X →* **R** satisfying the above two conditions corresponds to a
(not necessarily finite dimensional) feature representation in some feature space. [4] The corresponding feature space is a reproducing kernel 4 That is, there exists a feature transfor
Hilbert space (RKHS) which we will study in greater detail in sec- mation ***ϕ*** : **R** *[d]* *→* **R** *[e]* where *e* *∈* **R** *∪{* ∞ *}*

such that *k* ( ***x***, ***x*** *[′]* ) = ***ϕ*** ( ***x*** ) *[⊤]* ***ϕ*** ( ***x*** *[′]* ) . This is

tion 4.3.4. known as *Mercer’s theorem*, see theorem

4.2 of “Gaussian processes for machine

Intuitively, the kernel function evaluated at locations ***x*** and ***x*** *[′]* describes learning” (Williams and Rasmussen).


Hilbert space (RKHS) which we will study in greater detail in section 4.3.4.


Intuitively, the kernel function evaluated at locations ***x*** and ***x*** *[′]* describes
how *f* ( ***x*** ) and *f* ( ***x*** *[′]* ) are related. If ***x*** and ***x*** *[′]* are “close”, then *f* ( ***x*** )
and *f* ( ***x*** *[′]* ) are usually taken to be positively correlated, encoding a

“smooth” function.


*4.3.1* *Common Kernels*

We will now define some commonly used kernels. Often an additional
factor *σ* [2] ( *output scale* ) is added, which we assume to be 1 for simplicity.

1. The *linear kernel* is defined as

*k* ( ***x***, ***x*** *[′]* ; ***ϕ*** ) = . ***ϕ*** ( ***x*** ) *⊤* ***ϕ*** ( ***x*** *′* ) (4.16)

where ***ϕ*** is a nonlinear transformation as introduced in section 2.5
or the identity.


*f* ( *x* )

*x*

Figure 4.2: Functions sampled according
to a Gaussian process with a linear kernel and *ϕ* = id.

*f* ( *x* )

*x*

Figure 4.3: Functions sampled according to a Gaussian process with a linear
kernel and ***ϕ*** ( *x* ) = [ 1, *x*, *x* [2] ] (left) and
*ϕ* ( *x* ) = sin ( *x* ) (right).

2. The *Gaussian kernel* (also known as *squared exponential kernel* or *ra-*
*dial basis function (RBF) kernel* ) is defined as


�


*k* ( ***x***, ***x*** *[′]* ; *ℓ* ) = . exp


�


*−* *[∥]* ***[x]*** *[ −]* ***[x]*** *[′]* *[∥]* 2 [2]
2 *ℓ* [2]


(4.17)


gaussian processes 73

5 As the length scale is increased, the exponent of the exponential increases, resulting in a higher dependency between
locations.

Figure 4.4: Functions sampled according
to a Gaussian process with a Gaussian
kernel and length scales *ℓ* = 5 (left) and
*ℓ* = 1 (right).

*k* ( *x* *−* *x* *[′]* )

1.00

0.75

0.50

0.25

0.00

*−* 2 0 2

*x* *−* *x* *[′]*

Figure 4.5: Gaussian kernel with length
scales *ℓ* = 1, *ℓ* = 0.5, and *ℓ* = 0.2.

*k* ( *x* *−* *x* *[′]* )

1.00

0.75

0.50

0.25

0.00

*−* 2 0 2

*x* *−* *x* *[′]*

Figure 4.6: Laplace kernel with length
scales *ℓ* = 1, *ℓ* = 0.5, and *ℓ* = 0.2.


where *ℓ* is its *length scale* . The larger the length scale *ℓ*, the smoother
the resulting functions. [5]

*f* ( *x* )

*x*











3. The *Laplace kernel* (also known as *exponential kernel* ) is defined as


.
*k* ( ***x***, ***x*** *[′]* ; *ℓ* ) = exp *−* *[∥]* ***[x]*** *[ −]* ***[x]*** *[′]* *[∥]* [1]
� *ℓ*


. (4.18)
�


As can be seen in fig. 4.7, samples from a GP with Laplace kernel
are non-smooth as opposed to the samples from a GP with Gaus
sian kernel.

74 probabilistic artificial intelligence

*f* ( *x* )

*x*

4. The *Matérn kernel* trades the smoothness of the Gaussian and the
Laplace kernels. It is defined as


� *√*


Figure 4.7: Functions sampled according to a Gaussian process with a Laplace
kernel and length scales *ℓ* = 10 000 (left)
and *ℓ* = 10 (right).

6 Mean square continuity and mean
square differentiability are a generalization of continuity and differentiation
to random processes, where the limit
is generalized to a limit in the “mean
square sense”.
A sequence of random variables
*{* *X* *n* *}* *n* *∈* **N** is said to converge to the random variable *X* in *mean square* if

*n* lim *→* ∞ **[E]** � ( *X* *n* *−* *X* ) [2] [�] = 0. (4.20)

Using Markov’s inequality (1.81) it can
be seen that convergence in mean
square implies convergence in probability. Moreover, convergence in mean
square implies

*n* lim *→* ∞ **[E]** *[X]* *[n]* [ =] **[ E]** *[X]* and (4.21)

*n* lim *→* ∞ **[E]** *[X]* *n* [2] [=] **[ E]** *[X]* [2] [.] (4.22)

Whereas a deterministic function *f* ( ***x*** )
is said to be continuous at a point ***x*** *[⋆]* if
lim ***x*** *→* ***x*** *⋆* *f* ( ***x*** ) = *f* ( ***x*** *[⋆]* ), a random process
*f* ( ***x*** ) is *mean square continuous* at ***x*** *[⋆]* if

***x*** lim *→* ***x*** *[⋆]* **[E]** � ( *f* ( ***x*** ) *−* *f* ( ***x*** *[⋆]* )) [2] [�] = 0. (4.23)

It can be shown that a random process
is mean square continuous at ***x*** *[⋆]* iff its
kernel function *k* ( ***x***, ***x*** *[′]* ) is continuous at
***x*** = ***x*** *[′]* = ***x*** *[⋆]* .
Similarly, a random process *f* ( ***x*** ) is
*mean square differentiable* at a point ***x*** in
direction *i* if ( *f* ( ***x*** + *δ* ***e*** *i* ) *−* *f* ( ***x*** )) / *δ* converges in mean square as *δ* *→* 0 where
***e*** *i* is the unit vector in direction *i* . This
notion can be extended to higher-order
derivatives.
For the precise notions of mean square
continuity and mean square differentiability in the context of Gaussian processes, refer to section 4.1.1 of “Gaussian processes for machine learning”
(Williams and Rasmussen).


*k* ( ***x***, ***x*** *[′]* ; *ν*, *ℓ* ) = . 2 [1] *[−]* *[ν]*

Γ ( *ν* )


� *√*


2 *ν* *∥* ***x*** *−* ***x*** *[′]* *∥* 2

*ℓ*


*ν*

*K* *ν*
�


2 *ν* *∥* ***x*** *−* ***x*** *[′]* *∥* 2

*ℓ*


�

(4.19)


where Γ is the Gamma function, *K* *ν* the modified Bessel function
of the second kind, and *ℓ* a length scale parameter. The resulting functions are *⌈* *ν* *⌉−* 1 times mean square differentiable. [6] For

*ν* = [1] / 2, the Matérn kernel is equivalent to the Laplace kernel. For
*ν* *→* ∞, the Matérn kernel is equivalent to the Gaussian kernel. In
particular, GPs with a Gaussian kernel are infinitely many times
mean square differentiable whereas GPs with a Laplace kernel are
mean square continuous but not mean square differentiable.


gaussian processes 75

*4.3.2* *Kernel Composition*

Given kernels *k* 1 : *X × X →* **R** and *k* 2 : *X × X →* **R**, they can be
composed to obtain a new kernel *k* : *X × X →* **R** in the following

ways:

- *k* ( ***x***, ***x*** *[′]* ) = . *k* 1 ( ***x***, ***x*** *′* ) + *k* 2 ( ***x***, ***x*** *′* ),

- *k* ( ***x***, ***x*** *[′]* ) = . *k* 1 ( ***x***, ***x*** *′* ) *·* *k* 2 ( ***x***, ***x*** *′* ),

- *k* ( ***x***, ***x*** *[′]* ) = . *c* *·* *k* 1 ( ***x***, ***x*** *′* ) for any *c* *>* 0,

- *k* ( ***x***, ***x*** *[′]* ) = . *f* ( *k* 1 ( ***x***, ***x*** *′* )) for any polynomial *f* with positive coefficients or *f* = exp.

For example, the additive structure of a function *f* ( ***x*** ) = . *f* 1 ( ***x*** ) + *f* 2 ( ***x*** )
can be easily encoded in GP models. Suppose that *f* 1 *∼GP* ( *µ* 1, *k* 1 )
and *f* 2 *∼GP* ( *µ* 2, *k* 2 ), then the distribution of the sum of those two
functions *f* = *f* 1 + *f* 2 *∼GP* ( *µ* 1 + *µ* 2, *k* 1 + *k* 2 ) is another GP. [7] 7 We use *f* = . *f* 1 + *f* 2 to denote the function *f* ( *·* ) = *f* 1 ( *·* ) + *f* 2 ( *·* ) .
Whereas the addition of two kernels *k* 1 and *k* 2 can be thought of as
an *OR* operation (i.e., the kernel has high value if either *k* 1 or *k* 2 have
high value), the multiplication of *k* 1 and *k* 2 can be thought of as an
*AND* operation (i.e., the kernel has high value if both *k* 1 and *k* 2 have
high value). For example, the product of two linear kernels results in
functions which are quadratic.

As mentioned previously, the constant *c* of a scaled kernel function
*k* *[′]* ( ***x***, ***x*** *[′]* ) = . *c* *·* *k* ( ***x***, ***x*** *′* ) is generally called the *output scale* of a kernel,
and it scales the variance Var [ *f* ( ***x*** )] = *c* *·* *k* ( ***x***, ***x*** ) of the predictions *f* ( ***x*** )
from *GP* ( *µ*, *k* *[′]* ) .

*4.3.3* *Stationarity and Isotropy*

Kernel functions are commonly classified according to two properties.

**Definition 4.6** (Stationarity and isotropy) **.** A kernel *k* : **R** *[d]* *×* **R** *[d]* *→* **R**

is called

- *stationary* (or *shift-invariant* ) if there exists a function *k* [˜] such that

˜
*k* ( ***x*** *−* ***x*** *[′]* ) = *k* ( ***x***, ***x*** *[′]* ), and

- *isotropic* if there exists a function *k* [˜] such that *k* [˜] ( *∥* ***x*** *−* ***x*** *[′]* *∥* 2 ) = *k* ( ***x***, ***x*** *[′]* ) .

76 probabilistic artificial intelligence

Note that stationarity is a necessary condition for isotropy. In other
words, isotropy implies stationarity.



yes no *∥·∥* ***M*** denotes the Mahalanobis norm



*4.3.4* *Reproducing Kernel Hilbert Spaces*

Recall that Gaussian processes keep track of a posterior distribution
*f* *|* ***x*** 1: *n*, *y* 1: *n* over functions. It can be shown that the corresponding
MAP estimate *f* [ˆ] corresponds to a regularized optimization problem
in a suitable space of functions known as the *reproducing kernel Hilbert*
*space* . This duality is similar to the duality between the MAP estimate of Bayesian linear regression and ridge regression we observed
in chapter 2. So what is the reproducing kernel Hilbert space of a

kernel function *k* ?

**Definition 4.8** (Reproducing kernel Hilbert space, RKHS) **.** Given a kernel *k* : *X × X →* **R**, its corresponding *reproducing kernel Hilbert space* is
the space of functions *f* defined as


�


*H* *k* ( *X* ) = .


�


*n*
###### f ( x ) = ∑ α i k ( x, x i ) : n ∈ N, x i ∈X, α i ∈ R

*i* = 1


(4.27)


with inner product


*n*
###### ⟨ f, g ⟩ k = . ∑

*i* = 1


*n* *[′]*
###### ∑ α i α [′] j [k] [(] [x] [i] [,] [ x] [′] j [)] [,] (4.28)

*j* = 1

gaussian processes 77


where *g* ( ***x*** ) = ∑ *[n]* *j* = *[′]* 1 *[α]* *[′]* *j* *[k]* [(] ***[x]*** [,] ***[ x]*** *[′]* *j* [)] [ and norm] *[ ∥]* *[f]* *[ ∥]* *k* [=] �


*⟨* *f*, *f* *⟩* *k* .


It is straightforward to check that for any ***x*** *∈X*, *k* ( ***x***, *·* ) *∈H* *k* ( *X* ) . The
RKHS inner product *⟨·*, *·⟩* *k* satisfies for all ***x*** *∈X* and *f* *∈H* *k* ( *X* ) that
*f* ( ***x*** ) = *⟨* *f* ( *·* ), *k* ( ***x***, *·* ) *⟩* *k* which is also known as the *reproducing property* .

Foundational in characterizing the solution to regularized optimization problems within RKHSs is the *representer theorem* . [8] 8 Bernhard Schölkopf, Ralf Herbrich, and
Alex J Smola. A generalized represen
**Fact 4.10** (Representer theorem) **.** *Let k be a kernel and let λ* *>* 0 *. For* ter theorem. In *International conference on*

–

*f* *∈H* *k* ( *X* ) *and training data* *D* *comprising n observations, let Q* ( *f*, *D* ) *∈* *computational learning theory* 426. Springer, 2001, pages 416
**R** *∪{* ∞ *}* *denote any loss function. Then, any minimizer of*

*f* ˆ = . arg min *Q* ( *f*, *D* ) + *λ* *∥* *f* *∥* [2] *k* (4.29)
*f* *∈H* *k* ( *X* )

*admits a representation of the form*

*f* ˆ ( ***x*** ) = ˆ ***α*** *[⊤]* ***k*** ***x***, *A* *for some* ˆ ***α*** *∈* **R** *[n]* . (4.30)


78 probabilistic artificial intelligence



*4.4* *Model Selection*

We have not yet discussed how to pick the hyperparameters ***θ*** (e.g.,
parameters of kernels). A common technique in supervised learning
is to select hyperparameters ***θ***, such that the resulting function estimate

ˆ
*f* ***θ*** leads to the most accurate predictions on hold-out validation data.
After reviewing this approach, we contrast it with a Bayesian approach
to model selection, which avoids using point estimates of *f* [ˆ] ***θ*** and rather
utilizes the full posterior.

*4.4.1* *Optimizing Validation Set Performance*

A common approach to model selection is to split our data *D* into
separate training set *D* [train] = . *{* ( ***x*** train *i*, *y* [train] *i* ) *}* *i* *[n]* = 1 [and validation sets]
*D* [val] = . *{* ( ***x*** val *i*, *y* [val] *i* ) *}* *i* *[m]* = 1 [. We then optimize the model for a parameter]
candidate ***θ*** *j* using the training set. This is usually done by picking a
point estimate (like the MAP estimate),

*f* ˆ *j* = . arg max *p* ( *f* *|* ***x*** [train] 1: *n* [,] *[ y]* [train] 1: *n* [)] [.] (4.33)
*f*

Then, we score ***θ*** *j* according to the performance of *f* [ˆ] *j* on the validation
set,

ˆ
***θ*** = [.] arg max *p* ( *y* [val] 1: *m* *[|]* ***[ x]*** [val] 1: *m* [, ˆ] *[f]* *[j]* [)] [.] (4.34)
***θ*** *j*

This ensures that *f* [ˆ] *j* does not depend on *D* [val] .

gaussian processes 79





samples of the data are i.i.d.. Recall

ity (1.87) can be used to gauge how large
*m* should be.



While this approach often is quite effective at preventing overfitting
as compared to using the same data for training and picking ***θ*** [ˆ], it still
collapses the uncertainty in *f* into a point estimate. Can we do better?

*4.4.2* *Maximizing the Marginal Likelihood*

We have already seen for Bayesian linear regression, that picking a
point estimate loses a lot of information. Instead of optimizing the
effects of ***θ*** for a specific point estimate *f* [ˆ] of the model *f*, *maximizing*
*the marginal likelihood* optimizes the effects of ***θ*** across all realizations
of *f* . In this approach, we obtain our hyperparameter estimate via

***θ*** ˆ MLE = . arg max *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) (4.36) using the definition of marginal
***θ*** likelihood in Bayes’ rule (1.58)


= arg max
***θ***

= arg max
***θ***


*p* ( *y* 1: *n*, *f* *|* ***x*** 1: *n*, ***θ*** ) *d f* by conditioning on *f* using the sum rule
� (1.10)

*p* ( *y* 1: *n* *|* ***x*** 1: *n*, *f*, ***θ*** ) *p* ( *f* *|* ***θ*** ) *d f* . (4.37) using the product rule (1.14)
�


Remarkably, this approach typically avoids overfitting even though we
do not use a separate training and validation set. The following table provides an intuitive argument for why maximizing the marginal
likelihood is a good strategy.

likelihood prior

“underfit” model
small for “almost all” *f* large
(too simple ***θ*** )

“overfit” model large for “few” *f*
small
(too complex ***θ*** ) small for “most” *f*

“just right” moderate for “many” *f* moderate

For an “underfit” model, the likelihood is mostly small as the data
cannot be well described, while the prior is large as there are “fewer”
functions to choose from. For an “overfit” model, the likelihood is
large for “some” functions (which would be picked if we were only
minimizing the training error and not doing cross validation) but large


all possible data sets

Figure 4.8: A schematic illustration of
the marginal likelihood of a simple, intermediate, and complex model across
all possible data sets.


Table 4.1: The table gives an intuitive explanation of effects of parameter choices
***θ*** on the marginal likelihood. Note that
words in quotation marks refer to intuitive quantities, as we have infinitely
many realizations of *f* .



80 probabilistic artificial intelligence

for “most” functions. The prior is small, as the probability mass has
to be distributed among “more” functions. Thus, in both cases, one
term in the product will be small. Hence, maximizing the marginal
likelihood naturally encourages trading between a large likelihood and
a large prior.

In the context of Gaussian process regression, recall from eq. (4.3) that

*y* 1: *n* *|* ***x*** 1: *n*, ***θ*** *∼N* ( **0**, ***K*** *f*, ***θ*** + *σ* *n* [2] ***[I]*** [)] (4.38)

where ***K*** *f*, ***θ*** denotes the kernel matrix at the inputs ***x*** 1: *n* depending on
the kernel function parameterized by ***θ*** . We write ***K*** ***y***, ***θ*** = . ***K*** *f*, ***θ*** + *σ* *n* 2 ***[I]*** [.]
Continuing from eq. (4.36), we obtain

***θ*** ˆ MLE = arg max *N* ( ***y*** ; **0**, ***K*** ***y***, ***θ*** )
***θ***


(4.39) taking the negative logarithm
2 [log 2] *[π]*


=
arg min
***θ***

=
arg min
***θ***


1
2 ***[y]*** *[⊤]* ***[K]*** ***y*** *[−]*, ***θ*** [1] ***[y]*** [ +] [ 1]


*n*
2 [log det] � ***K*** ***y***, ***θ*** � +


1
2 ***[y]*** *[⊤]* ***[K]*** ***y*** *[−]*, ***θ*** [1] ***[y]*** [ +] [ 1] 2 [log det] � ***K*** ***y***, ***θ*** � (4.40) the last term is independent of ***θ***


The first term of the optimization objective describes the “goodness of
fit” (i.e., the “alignment” of ***y*** with ***K*** ***y***, ***θ*** ). The second term characterizes the “volume” of the model class. Thus, this optimization naturally
trades the aforementioned objectives.

Marginal likelihood maximization is an empirical Bayes method. Often
it is simply referred to as *empirical Bayes* . It also has the nice property
that the gradient of its objective (the MLL loss) can be expressed in
closed-from,


�


*∂* log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) = [1]
*∂θ* 2 [tr]
*j*


�


( ***αα*** *[⊤]* *−* ***K*** ***y*** *[−]*, ***θ*** [1] [)] *[∂]* ***[K]*** *∂θ* ***[y]*** [,] ***[θ]***

*j*


(4.41)


where ***α*** = . ***K*** *−* 1
***y***, ***θ*** ***[y]*** [ and tr] [(] ***[M]*** [)] [ is the trace of a matrix] ***[ M]*** [. This optimiza-]
tion problem is, in general, non-convex. Figure 4.9 gives an example
of two local optima according to empirical Bayes.

10 [1]

10 [0]

10 *[−]* [1]

2

1

0

*−* 1

*−* 2


10 [0] 10 [1]

lengthscale *h*

2

1

0

*−* 1

*−* 2


gaussian processes 81

Figure 4.9: The top plot shows contour
lines of an empirical Bayes with two local optima. The bottom two plots show
the Gaussian processes corresponding
to the two optimal models. The left
model with smaller lengthscale is chosen
within a more flexible class of models,
while the right model explains more observations through noise. Adapted from
figure 5.5 of “Gaussian processes for
machine learning” (Williams and Rasmussen).


*−* 5.0 *−* 2.5 0.0 2.5 5.0

*x*


*−* 5.0 *−* 2.5 0.0 2.5 5.0

*x*

82 probabilistic artificial intelligence




1.0

0.5

0.0

0 100 200

# of iterations

Figure 4.10: An example of model selection by maximizing the log likelihood
(without hyperpriors) using a **linear**,
**quadratic**, **Laplace**, **Matérn** ( *ν* = 3 / 2 ),
and **Gaussian** kernel, respectively. They
are used to learn the function


*x* *�→* [sin] [(] *[x]* [)] + *ϵ*, *ϵ* *∼N* ( 0, 0.01 )

*x*



using SGD with learning rate 0.1.








Taking a step back, observe that taking a Bayesian perspective on
model selection naturally led us to consider all realizations of our
model *f* instead of using point estimates. However, we are still using point estimates for our model parameters ***θ*** . Continuing on our
Bayesian adventure, we could place a prior *p* ( ***θ*** ) on them too. [11] We 11 Such a prior is called *hyperprior* .
could use it to obtain the MAP estimate (still a point estimate!) which
adds an additional regularization term

***θ*** ˆ MAP = . arg max *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) (4.44)
***θ***

= arg min *−* log *p* ( ***θ*** ) *−* log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) . (4.45) using Bayes’ rule (1.58) and then taking
***θ*** the negative logarithm

An alternative approach is to consider the full posterior distribution
over parameters ***θ*** . The resulting predictive distribution is, however,

gaussian processes 83


intractable,

*p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ) = *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, *f* ) *·* *p* ( *f* *|* ***x*** 1: *n*, *y* 1: *n*, ***θ*** ) *·* *p* ( ***θ*** ) *d f d* ***θ*** .
��

(4.46)

Recall that as the mode of Gaussians coincides with their mean, the

MAP estimate corresponds to the mean of the predictive posterior.

As a final note, observe that in principle, there is nothing stopping us
from descending deeper in the Bayesian hierarchy. The prior on the
model parameters ***θ*** is likely to have parameters too. Ultimately, we
need to break out of this hierarchy of dependencies and choose a prior.

*4.5* *Approximations*

To learn a Gaussian process, we need to invert matrices, hence the
computational cost is *O* � *n* [3] [�] . Compare this to Bayesian linear regression which allows us to learn a regression model in *O* � *nd* [2] [�] time (even
online) where *d* is the feature dimension. It is therefore natural to look
for ways of approximating a Gaussian process.

*4.5.1* *Local Methods*

Recall that during forward sampling, we had to condition on a larger
and larger number of previous samples. When sampling at a location
***x***, a very simple approximation is to only condition on those samples
***x*** *[′]* that are “close” (where *|* *k* ( ***x***, ***x*** *[′]* ) *| ≥* *τ* for some *τ* *>* 0). Essentially,
this method “cuts off the tails” of the kernel function *k* . However, *τ*

has to be chosen carefully as if *τ* is chosen too large, samples become
essentially independent.

This is one example of a *sparse approximation* of a Gaussian process. We
will discuss more advanced sparse approximations known as “inducing point methods” in section 4.5.3.

*4.5.2* *Kernel Function Approximation*

Another method is to approximate the kernel function directly. The
idea is to construct a “low-dimensional” feature map ***ϕ*** : **R** *[d]* *→* **R** *[m]*

that approximates the kernel,

*k* ( ***x***, ***x*** *[′]* ) *≈* ***ϕ*** ( ***x*** ) *[⊤]* ***ϕ*** ( ***x*** *[′]* ) . (4.47)

84 probabilistic artificial intelligence

Then, we can apply Bayesian linear regression, resulting in a time complexity of *O* � *nm* [2] + *m* [3] [�] .

One example of this approach are *random Fourier features*, which we
will discuss in the following.


*i*

0


Im




|m|Col2|Col3|
|---|---|---|
|eiφ sin φ φ cos φ|||
||||








Because a stationary kernel *k* : **R** *[d]* *×* **R** *[d]* *→* **R** can be interpreted as a
function in one variable, it has an associated Fourier transform which
we denote by *p* ( ***ω*** ) . That is,


0 1

Re

Figure 4.11: Illustration of Euler’s formula. It can be seen that *e* *[i][φ]* corresponds
to a (counter-clockwise) rotation on the
unit circle as *φ* varies from 0 to 2 *π* .

*f* ( *x* )

1

0

*−* 1 1

*x*

ˆ
*f* ( *ω* )

2

0

*−* *π* *π*

*ω*

Figure 4.12: The Fourier transform of a
rectangular pulse,


1 *x* *∈* [ *−* 1, 1 ]
�0 otherwise,


*k* ( ***x*** *−* ***x*** *[′]* ) = (4.51)
� **R** *[d]* *[ p]* [(] ***[ω]*** [)] *[e]* *[i]* ***[ω]*** *[⊤]* [(] ***[x]*** *[−]* ***[x]*** *[′]* [)] *[ d]* ***[ω]*** [.]


*f* ( *x* ) = .

is given by


**Fact 4.15** (Bochner’s theorem) **.** *A continuous stationary kernel on* **R** *[d]* *is*
*positive definite if and only if its Fourier transform p* ( ***ω*** ) *is non-negative.*


ˆ 1
*f* ( *ω* ) = � *−*


= [2 sin] *ω* [(] *[ω]* [)] .



[1]
*−* 1 *[e]* *[−]* *[i][ω][x]* *[ dx]* [ =] *iω*


*iω*


� *e* *[i][ω]* *−* *e* *[−]* *[i][ω]* [�]

gaussian processes 85


Bochner’s theorem implies that when a continuous and stationary kernel is positive definite and scaled appropriately, its Fourier transform
*p* ( ***ω*** ) is a proper probability distribution. In this case, *p* ( ***ω*** ) is called
the *spectral density* of the kernel *k* .





using the definition of the Fourier


*d* ***x*** using the definition of the Gaussian
kernel (4.17)









The key idea is now to interpret the kernel as an expectation,


*k* ( ***x*** *−* ***x*** *[′]* ) =
�


from eq. (4.51)
**R** *[d]* *[ p]* [(] ***[ω]*** [)] *[e]* *[i]* ***[ω]*** *[⊤]* [(] ***[x]*** *[−]* ***[x]*** *[′]* [)] *[ d]* ***[ω]***


= **E** ***ω*** *∼* *p* � *e* *[i]* ***[ω]*** *[⊤]* [(] ***[x]*** *[−]* ***[x]*** *[′]* [)] [�] by the definition of expectation (1.23b)

= **E** ***ω*** *∼* *p* �cos ( ***ω*** *[⊤]* ***x*** *−* ***ω*** *[⊤]* ***x*** *[′]* ) + *i* sin ( ***ω*** *[⊤]* ***x*** *−* ***ω*** *[⊤]* ***x*** *[′]* ) �. using Euler’s formula (4.48)

Observe that as both *k* and *p* are real, convergence of the integral implies **E** ***ω*** *∼* *p* �sin ( ***ω*** *[⊤]* ***x*** *−* ***ω*** *[⊤]* ***x*** *[′]* ) � = 0. Hence,

= **E** ***ω*** *∼* *p* �cos ( ***ω*** *[⊤]* ***x*** *−* ***ω*** *[⊤]* ***x*** *[′]* ) �

= **E** ***ω*** *∼* *p* **E** *b* *∼* Unif ([ 0,2 *π* ]) �cos (( ***ω*** *[⊤]* ***x*** + *b* ) *−* ( ***ω*** *[⊤]* ***x*** *[′]* + *b* )) � epanding with *b* *−* *b*

= **E** ***ω*** *∼* *p* **E** *b* *∼* Unif ([ 0,2 *π* ]) �cos ( ***ω*** *[⊤]* ***x*** + *b* ) cos ( ***ω*** *[⊤]* ***x*** *[′]* + *b* ) using the angle subtraction identity,

cos ( *α* *−* *β* ) = cos *α* cos *β* + sin *α* sin *β*
+ sin ( ***ω*** *[⊤]* ***x*** + *b* ) sin ( ***ω*** *[⊤]* ***x*** *[′]* + *b* )
�

86 probabilistic artificial intelligence

= **E** ***ω*** *∼* *p* **E** *b* *∼* Unif ([ 0,2 *π* ]) �2 cos ( ***ω*** *[⊤]* ***x*** + *b* ) cos ( ***ω*** *[⊤]* ***x*** *[′]* + *b* ) � using

= **E** ***ω*** *∼* *p*, *b* *∼* Unif ([ 0,2 *π* ]) � *z* ***ω***, *b* ( ***x*** ) *·* *z* ***ω***, *b* ( ***x*** *[′]* ) � (4.53)


**E** *b* [ cos ( *α* + *b* ) cos ( *β* + *b* )]

= **E** *b* [ sin ( *α* + *b* ) sin ( *β* + *b* )]


where *z* ***ω***, *b* ( ***x*** ) = . *√*


2 cos ( ***ω*** *[⊤]* ***x*** + *b* ),


for *b* *∼* Unif ([ 0, 2 *π* ])


*≈* [1]

*m*


*m*
###### ∑ z ω ( i ), b ( i ) ( x ) · z ω ( i ), b ( i ) ( x [′] ) (4.54) using Monte Carlo sampling to estimate

*i* = 1 the expectation, see example 1.38


for independent samples ***ω*** [(] *[i]* [)] [ iid] *∼* *p* and *b* [(] *[i]* [)] [ iid] *∼* Unif ([ 0, 2 *π* ]),

= ***z*** ( ***x*** ) *[⊤]* ***z*** ( ***x*** *[′]* ) (4.55)

where

***z*** ( ***x*** ) = . *√* 1 *m* [ *z* ***ω*** ( 1 ), *b* ( 1 ) ( ***x*** ), . . ., *z* ***ω*** ( *m* ), *b* ( *m* ) ( ***x*** )] *[⊤]* (4.56)

is the (randomized) feature map of random Fourier features.

Intuitively, each component of the feature map ***z*** ( ***x*** ) projects ***x*** onto a
random direction ***ω*** drawn from the (inverse) Fourier transform *p* ( ***ω*** )
of *k* ( ***x*** *−* ***x*** *[′]* ), and wraps this line onto the unit circle in **R** [2] . After transforming two points ***x*** and ***x*** *[′]* in this way, their inner product is an unbiased estimator of *k* ( ***x*** *−* ***x*** *[′]* ) . The mapping *z* ***ω***, *b* ( ***x*** ) = *√* 2 cos ( ***ω*** *[⊤]* ***x*** + *b* )

additionally rotates the circle by a random amount *b* and projects the
points onto the interval [ 0, 1 ] .

Rahimi et al. show that Bayesian linear regression with the feature
map ***z*** approximates Gaussian processes with a stationary kernel: [12]

**Theorem 4.18** (Uniform convergence of Fourier features) **.** *Suppose* *M*
*is a compact subset of* **R** *[d]* *with diameter* diam ( *M* ) *. Then for a stationary*
*kernel k, the random Fourier features* ***z*** *, and any ϵ* *>* 0 *it holds that*


4

2

0

2

1

0

*−* 1


*f* ( *x* )

|Col1|Col2|
|---|---|
|||
|||
|||
||−5.0 −2.5 0.0|
|||
|||
|||
|||



*−* 5.0 *−* 2.5 0.0 2.5 5.0

*x*


�


Figure 4.13: Example of random Fourier
features with where the number of fea
tures *m* is 5 (top) and 10 (bottom), respectively. The noise-free true function
is shown in black and the mean of the

Gaussian process is shown in blue.

12 Ali Rahimi, Benjamin Recht, et al. Random features for large-scale kernel machines. In *NIPS*, 2007


**P**


sup

� ***x***, ***x*** *[′]* *∈M*


*⊤* *′* *′*
***z*** ( ***x*** ) ***z*** ( ***x*** ) *−* *k* ( ***x*** *−* ***x*** ) *≥* *ϵ*
��� ���


(4.57)


�


*≤* 2 [8] *σ* *p* diam ( *M* )
� *ϵ*


2
*mϵ* [2]
exp *−*
� � 8 ( *d* + 2 )


*where σ* *p* [2] = . **E** ***ω*** *∼* *p* � ***ω*** *[⊤]* ***ω*** � *is the second moment of p, m is the dimension of*
***z*** ( ***x*** ) *, and d is the dimension of* ***x*** *.*

Note that the error probability decays exponentially fast in the dimension of the Fourier feature space.

gaussian processes 87

















88 probabilistic artificial intelligence



cess



ing and smoothing solutions to tempo
(Hartikainen and Särkkä).



*f* ( *x* )


*4.5.3* *Data Sampling*


2

1


Another natural approach is to only consider a (random) subset of the 0
training data during learning. The naïve approach is to subsample
uniformly at random. Not very surprisingly, we can do much better. *−* 5 0 5

*x*


One subsampling method is the so-called *inducing points method* . The
idea is to summarize the data around so-called inducing points. [15] For

now, let us consider an arbitrary set of inducing points,

*U* = . *{* ***x*** 1, . . ., ***x*** *k* *}* .

Then, the original Gaussian process can be recovered using marginal
ization,


Figure 4.14: Inducing points ***u*** are shown
as vertical dotted red lines. The noise
free true function is shown in black and
the mean of the Gaussian process is
shown in blue. Observe that the true
function is approximated “well” around
the inducing points.

15 The inducing points can be treated as
hyperparameters.


*p* ( *f* *[⋆]*, ***f*** ) =
�


**R** *[k]* *[ p]* [(] *[ f]* *[ ⋆]* [,] ***[ f]*** [,] ***[ u]*** [)] *[ d]* ***[u]*** [ =] �


(4.59) using the sum rule (1.10) and product
**R** *[k]* *[ p]* [(] *[ f]* *[ ⋆]* [,] ***[ f]*** *[ |]* ***[ u]*** [)] *[p]* [(] ***[u]*** [)] *[ d]* ***[u]*** [,] rule (1.14)


where ***f*** = [ . *f* ( ***x*** 1 ) *· · ·* *f* ( ***x*** *n* )] *[⊤]* and *f* *[⋆]* = . *f* ( ***x*** *⋆* ) at some evaluation
. *⊤* *k*
point ***x*** *[⋆]* *∈X* . We use ***u*** = [ *f* ( ***x*** 1 ) *· · ·* *f* ( ***x*** *k* )] *∈* **R** to denote the predictions of the model at the inducing points *U* . Due to the marginalization property of Gaussian processes (4.1), we have that ***u*** *∼N* ( **0**, ***K*** *UU* ) .
The key idea is to approximate the joint prior, assuming that *f* *[⋆]* and ***f***

gaussian processes 89


are conditionally independent given ***u***,


*p* ( *f* *[⋆]*, ***f*** ) *≈* *q* ( *f* *[⋆]*, ***f*** ) = (4.60)
� **R** *[k]* *[ q]* [(] *[ f]* *[ ⋆]* *[|]* ***[ u]*** [)] *[q]* [(] ***[ f]*** *[ |]* ***[ u]*** [)] *[p]* [(] ***[u]*** [)] *[ d]* ***[u]*** [.]


Here, *q* ( ***f*** *|* ***u*** ) and *q* ( *f* *[⋆]* *|* ***u*** ) are approximations of the *training condi-*
*tional p* ( ***f*** *|* ***u*** ) and the *testing conditional p* ( *f* *[⋆]* *|* ***u*** ), respectively. Still
writing *A* = *{* ***x*** 1, . . ., ***x*** *n* *}* and defining *⋆* = . *{* ***x*** *⋆* *}*, we know, using the
closed-form expression for conditional Gaussians (1.117),

*q* ( ***f*** *|* ***u*** ) *∼N* ( ***f*** ; ***K*** *AU* ***K*** *UU* *[−]* [1] ***[u]*** [,] ***[ K]*** *[AA]* *[ −]* ***[Q]*** *[AA]* [)] [,] (4.61a)

*q* ( *f* *[⋆]* *|* ***u*** ) *∼N* ( *f* *[⋆]* ; ***K*** *⋆* *U* ***K*** *UU* *[−]* [1] ***[u]*** [,] ***[ K]*** *[⋆⋆]* *[−]* ***[Q]*** *[⋆⋆]* [)] (4.61b)

where ***Q*** *ab* = . ***K*** *aU* ***K*** *UU* *−* 1 ***[K]*** *[Ub]* [. Intuitively,] ***[ K]*** *[AA]* [ represents the prior co-]
variance and ***Q*** *AA* represents the covariance “explained” by the inducing points. [16] 16 For more details, refer to section 2
of “A unifying view of sparse apComputing the full covariance matrix is expensive. In the following, proximate Gaussian process regression”
(Quinonero-Candela and Rasmussen).
we mention two approximations to the covariance of the training conditional (and testing conditional).










90 probabilistic artificial intelligence



*f* ( *x* )

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|||||||
||−5.0||−2.5 0.|0 2.5 5.0||
|||||||
|||||||
|||||||



*−* 5.0 *−* 2.5 0.0 2.5 5.0

*x*


The computational cost for inducing point methods SoR and FITC is
dominated by the cost of inverting ***K*** *UU* . Thus, the time complexity is
cubic in the number of inducing points, but only linear in the number
of data points.


3

2

1

0

*−* 1

2

0

*−* 2


Figure 4.15: Comparison of SoR (top)
and FITC (bottom). The inducing points
***u*** are shown as vertical dotted red lines.

The noise-free true function is shown in
black and the mean of the Gaussian process is shown in blue.

### *5* *Variational Inference*

We have seen how to do learning and inference with Gaussians, exploiting their closed-form formulas for marginal and conditional dis
tributions. But what if we work with other distributions?

In this and the following chapter, we will discuss two methods of approximate inference. We begin by discussing variational inference,
which aims to find a good approximation of the posterior distribution from which it is easy to sample. In chapter 6, we discuss Markov
chain Monte Carlo methods, which approximate the sampling from
the posterior distribution directly.

The fundamental idea behind variational inference is to approximate
the true posterior distribution using a “simpler” posterior that is as
close as possible to the true posterior.

*p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) = [1] = . *q* ***λ*** ( ***θ*** ) (5.1)

*Z* *[p]* [(] ***[θ]*** [,] *[ y]* [1:] *[n]* *[ |]* ***[ x]*** [1:] *[n]* [)] *[ ≈]* *[q]* [(] ***[θ]*** *[ |]* ***[ λ]*** [)]

where ***λ*** represents the parameters of the *variational posterior q*, also
called *variational parameters* . In doing so, variational inference reduces
learning and inference (where the fundamental difficulty lies in solving high-dimensional integrals) to an optimization problem. We have
already seen in section 1.5 that optimizing (stochastic) objectives is a
well-understood problem with efficient algorithms that perform well
in practice.

*5.1* *Laplace Approximation*

A natural idea for approximating the posterior distribution is to use
a Gaussian approximation (that is, a second-order Taylor approximation) around the mode of the posterior. Such an approximation is
known as a Laplace approximation.

92 probabilistic artificial intelligence

Let

*ψ* ( ***θ*** ) = . log *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) (5.2)

denote the log-posterior. Then, using a second-order Taylor approximation (1.103) around the mode ***θ*** [ˆ] of *ψ* (i.e., the MAP estimate), we

obtain

*ψ* ( ***θ*** ) *≈* *ψ* [ˆ] ( ***θ*** ) = . *ψ* ( ˆ ***θ*** ) + ( ***θ*** *−* ***θ*** ˆ ) *⊤* ***∇*** *ψ* ( ˆ ***θ*** ) + 1

2 [(] ***[θ]*** *[ −]* ***[θ]*** [ˆ] [)] *[⊤]* ***[H]*** *[ψ]* [(] [ ˆ] ***[θ]*** [)(] ***[θ]*** *[ −]* ***[θ]*** [ˆ] [)]

= *ψ* ( ***θ*** [ˆ] ) + [1] (5.3) using ***∇*** *ψ* ( ***θ*** [ˆ] ) = 0

2 [(] ***[θ]*** *[ −]* ***[θ]*** [ˆ] [)] *[⊤]* ***[H]*** *[ψ]* [(] [ ˆ] ***[θ]*** [)(] ***[θ]*** *[ −]* ***[θ]*** [ˆ] [)] [.]

Compare this expression to the log of the PDF of a Gaussian,

log *N* ( ***θ*** ; ***θ*** [ˆ], ***Λ*** *[−]* [1] ) = *−* [1] (5.4)

2 [(] ***[θ]*** *[ −]* ***[θ]*** [ˆ] [)] *[⊤]* ***[Λ]*** [(] ***[θ]*** *[ −]* ***[θ]*** [ˆ] [) +] [ const.]

Thus,

*ψ* ˆ ( ***θ*** ) = *ψ* ( ˆ ***θ*** ) + [1]

2 [(] ***[θ]*** *[ −]* ***[θ]*** [ˆ] [)] *[⊤]* ***[H]*** *[ψ]* [(] [ ˆ] ***[θ]*** [)(] ***[θ]*** *[ −]* ***[θ]*** [ˆ] [)]

= log *N* ( ***θ*** ; ***θ*** [ˆ], *−* ***H*** *ψ* ( ***θ*** [ˆ] ) *[−]* [1] ) + const, (5.5)

where we note that *ψ* ( ***θ*** [ˆ] ) is a constant. We therefore define

***Λ*** = [.] *−* ***H*** *ψ* ( ***θ*** [ˆ] ) = *−* ***H*** ***θ*** log *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) �� ***θ*** = ***θ*** [ˆ] [.] (5.6)

Observe that ***Λ*** is symmetric. Moreover, *ψ* is a concave function and
hence ***H*** *ψ* ( ***θ*** ) is negative semi-definite, [1] implying that ***Λ*** is positive 1 see remark 1.58
semi-definite. Using that the inverse of a positive semi-definite matrix
is also positive semi-definite, ***Λ*** *[−]* [1] is a valid covariance matrix. The
*Laplace approximation* of *p* is then defined as

*q* ( ***θ*** ) = . *N* ( ***θ*** ; ˆ ***θ***, ***Λ*** *−* 1 ) ∝ exp ( ˆ *ψ* ( ***θ*** )) . (5.7)



of ***Σ***


variational inference 93

*x* 2

10

5

0

*−* 5

*−* 2 0 2

*x* 1

Figure 5.1: Logistic regression classifies
data into two classes with a linear decision boundary.


*Gaussian process classification* . See exercise 5.6 and “Scalable variational Gaus
et al.).










*σ* ( *z* )

1.00

0.75

0.50

0.25

0.00

*−* 5 0 5

*z*

Figure 5.2: The logistic function
squashes the linear function ***w*** *[⊤]* ***x*** onto
the interval ( 0, 1 ) .






94 probabilistic artificial intelligence





variational inference 95

0.8



0.6

0.4

0.2

0.0


|Col1|q pˆ p|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||


*−* 2 0 2 4 6

*x*


Intuitively, the Laplace approximation matches the shape of the true
posterior around its mode but may not represent it accurately elsewhere — often leading to extremely overconfident predictions.


Figure 5.3: The Laplace approximation
*q* greedily selects the mode of the true
posterior distribution *p* and matches the
curvature around the mode ˆ *p* . As shown
here, the Laplace approximation can be
extremely overconfident when *p* is not
approximately Gaussian.






*5.2* *Inference with a Variational Posterior*

How can we perform inference using our variational approximation?
We simply approximate the (intractable) true posterior with our variational posterior,

*p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ) = *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ) *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) *d* ***θ*** using the sum rule (1.10)
�

*≈* *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ) *q* ***λ*** ( ***θ*** ) *d* ***θ*** (5.21)
�

96 probabilistic artificial intelligence

At first sight, it may seem as though we did not gain much by our
approximation. However, observe that interpreting the integral from
a function-space view, the final prediction *y* *[⋆]* is conditionally independent from the model parameters ***θ*** given the prediction from the model
*f* *[⋆]* : [3] 3 This is discussed in greater detail in
section 2.6.

= *p* ( *y* *[⋆]* *|* *f* *[⋆]* ) *p* ( *f* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ) *q* ***λ*** ( ***θ*** ) *d* ***θ*** *d f* *[⋆]* once more, using the sum rule (1.10)
��

= *p* ( *y* *[⋆]* *|* *f* *[⋆]* ) *p* ( *f* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ) *q* ***λ*** ( ***θ*** ) *d* ***θ*** *d f* *[⋆]* rearranging terms
� �

= *p* ( *y* *[⋆]* *|* *f* *[⋆]* ) *q* ***λ*** ( *f* *[⋆]* *|* ***x*** *[⋆]* ) *d f* *[⋆]* . (5.22) marginalizing out ***θ*** using the sum rule
� (1.10)

Generally, *f* *[⋆]* = *g* ( ***x*** *[⋆]* ; ***θ*** ) where *g* is a deterministic function representing our “model” parameterized by ***θ*** . Then, *p* ( *f* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ) = *δ* *g* ( ***x*** *⋆* ; ***θ*** ) ( *f* *[⋆]* )
is a point density (cf. example 1.20) and *q* ***λ*** ( *f* *[⋆]* *|* ***x*** *[⋆]* ) is obtained from
*q* ***λ*** ( ***θ*** ) using a “change of variables” (cf. section 1.1.11).

We have replaced the (high-dimensional) integral over the model parameters ***θ*** by the (one-dimensional) integral over the prediction of
our variational posterior *f* *[⋆]* . While this integral is generally still intractable, it can be approximated efficiently using numerical quadrature methods such as the Gauss-Legendre quadrature.

variational inference 97




98 probabilistic artificial intelligence








*5.3* *Information Theory*

One of our main objectives throughout this course is to capture the
“uncertainty” about events *A* in an appropriate probability space. One
very natural measure of uncertainty is their probability, **P** ( *A* ) . In
this section, we will introduce an alternative measure of uncertainty,
namely the so-called “suprise” about the event *A* .

*5.3.1* *Surprise*

The *surprise* about an event with probability *u* is defined as

S [ *u* ] = . *−* log *u* . (5.28)

Observe that the surprise is a function from **R** *≥* 0 to **R**, where we let
S [ 0 ] *≡* ∞. Moreover, for a discrete random variable *X*, we have that


4

2

0


S [ *u* ]

0.00 0.25 0.50 0.75 1.00

*u*


Figure 5.4: Surprise S [ *u* ] associated with
an event of probability *u* .

variational inference 99

*p* ( *x* ) *≤* 1, and hence, S [ *p* ( *x* )] *≥* 0. But why is it reasonable to measure
surprise by *−* log *u* ?

Remarkably, it can be shown that the following natural axiomatic characterization leads to exactly this definition of surprise and entropy.

**Theorem 5.7** (Axiomatic characterization of surprise) **.** *The axioms*

*1.* S [ 1 ] = 0 *certain events are not surprising*
*2.* S [ *u* ] *>* S [ *v* ] = *⇒* *u* *<* *v (anti-monotonicity)* *we are more surprised by unlikely events*
*3.* S *is continuous,* *no jumps in surprise for infinitesimal*
*4.* S [ *uv* ] = S [ *u* ] + S [ *v* ] *for independent events,* *changes of probability*

*characterize* S *up to a positive constant factor.*

*Proof.* Cauchy’s functional equation, *f* ( *x* + *y* ) = *f* ( *x* ) + *f* ( *y* ), has the
unique family of solutions *{* *f* : *x* *�→* *cx* : *c* *∈* **R** *}* if *f* is required to be

continuous. Such a solution is called an “additive function”. Consider

the function *g* ( *x* ) = . *f* ( *e* *x* ) . Then, *g* is additive if and only if

*f* ( *e* *[x]* *e* *[y]* ) = *f* ( *e* *[x]* [+] *[y]* ) = *g* ( *x* + *y* ) = *g* ( *x* ) + *g* ( *y* ) = *f* ( *e* *[x]* ) + *f* ( *e* *[y]* ) .

It therefore follows from the third and fourth axioms of surprise that
S [ *u* ] = *c* log *u* for any *c* *∈* **R** . The second axiom of surprise implies that
*c* *<* 0, and thus, S [ *u* ] = *−* *c* *[′]* log *u* for any *c* *[′]* *>* 0. Finally, observe that
the first axiom of surprise is satisfied for any such *c* *[′]* .


Importantly, surprise offers a different perspective on uncertainty as
opposed to probability: the uncertainty about an event can either be
interpreted in terms of its probability or in terms of its surprise, and
the two “spaces of uncertainty” are related by a log-transform. This
relationship is illustrated in fig. 5.5. Information theory is the study of
uncertainty in terms of surprise.

Throughout this course we will see many examples where modeling
uncertainty in terms of surprise (i.e., the information-theoretic interpretation of uncertainty) is useful. One example where we have already encountered the “surprise space” was in the context of likelihood maximization (cf. section 1.4.3) where we used that the logtransform linearizes products of probabilities. We will see later in section 6.3 that in many cases the surprise S [ *p* ( *x* )] can also be interpreted
as a “cost” or “energy” associated with the state *x* .

*5.3.2* *Entropy*

The *entropy* of a distribution *p* is the average surprise about samples
from *p* . In this way, entropy is a notion of uncertainty associated with




Figure 5.5: Illustration of the probability
space and the corresponding “surprise
space”.

100 probabilistic artificial intelligence

the distribution *p* : if the entropy of *p* is large, we are more uncertain
about *x* *∼* *p* than if the entropy of *p* were low. Formally,

.
H [ *p* ] = **E** *x* *∼* *p* [ S [ *p* ( *x* )]] = **E** *x* *∼* *p* [ *−* log *p* ( *x* )] . (5.29)

When **X** *∼* *p* is a random vector distributed according to *p*, we write
H [ **X** ] = . H [ *p* ] . Observe that by definition, if *p* is discrete then H [ *p* ] *≥* 0
as *p* ( *x* ) *≤* 1 ( *∀* *x* ) . [4] For discrete distributions it is common to use 4 The entropy of a continuous distribu
the logarithm with base 2 rather than the natural logarithm. [5] Thus, tion can be negative. For example,


entropy is given by
###### H [ p ] = − ∑ p ( x ) log 2 p ( x ) (if p is discrete), (5.30a)

*x*

H [ *p* ] = *−* *p* ( ***x*** ) log *p* ( ***x*** ) *d* ***x*** (if *p* is continuous). (5.30b)
�


1 1
H [ Unif ([ *a*, *b* ])] = *−*
� *b* *−* *a* [log] *b* *−* *a* *[dx]*

= log ( *b* *−* *a* )

which is negative if *b* *−* *a* *<* 1.

5 Recall that log 2 *x* = [lo] log 2 [g] *[ x]* [, that is, loga-]

rithms to a different base only differ by
a constant factor.

*H* [ Bern ( *p* )]

1.00

0.75

0.50

0.25

0.00

0.0 0.5 1.0

*p*

Figure 5.6: Entropy H [ Bern ( *p* )] of a Bernoulli experiment with success probability *p* .










variational inference 101



















using **E** � ( *x* *−* *µ* ) 2 � = Var [ *x* ] = *σ* 2 (1.44)







Observe that the surprise S [ *u* ] is convex in *u* . Jensen’s inequality is a
useful tool when working with expected surprise (i.e., entropy).


|g [g(X)] E g( [X]) E|Col2|
|---|---|
|||


**E** [ *X* ]

Figure 5.7: An illustration of Jensen’s inequality. Due to the convexity of *g*, we
have that *g* evaluated at **E** [ *X* ] will always
be below the average of evaluations of *g* .




102 probabilistic artificial intelligence



From the fourth axiom of surprise, we immediately obtain the following property of entropy.

**Corollary 5.13.** *For a multivariate distribution p that can be factorized into*
*p* ( *x* 1: *d* ) = ∏ *i* *[d]* = 1 *[p]* *[i]* [(] *[x]* *[i]* [)] *[, we have that]*

*d*
###### H [ p ] = ∑ H [ p i ] . (5.36)

*i* = 1

*5.3.3* *Cross-Entropy*

How can we use entropy to measure our average surprise when assuming the data follows some distribution *q* but in reality the data
follows a different distribution *p* ?

**Definition 5.14** (Cross-entropy) **.** The *cross-entropy* of a distribution *q*
relative to the distribution *p* is

.
H [ *p* *∥* *q* ] = **E** *x* *∼* *p* [ S [ *q* ( *x* )]] = **E** *x* *∼* *p* [ *−* log *q* ( *x* )] . (5.37)

Cross-entropy can also be expressed in terms of the KL-divergence (cf.
section 5.5) KL ( *p* *∥* *q* ) which measures how “different” the distribution
*q* is from a reference distribution *p*,

H [ *p* *∥* *q* ] = H [ *p* ] + KL ( *p* *∥* *q* ) *≥* H [ *p* ] . (5.38) KL ( *p* *∥* *q* ) *≥* 0 is shown in exercise 5.20

Quite intuitively, the average surprise in samples from *p* with respect
to the distribution *q* is given by the inherent uncertainty in *p* and the
additional surprise that is due to us assuming the wrong data distribution *q* . The “closer” *q* is to the true data distribution *p*, the smaller
is the additional average surprise.


variational inference 103








*5.4* *Variational Families*

Laplace approximation approximates the true (intractable) posterior
with a simpler one, by greedily matching mode and curvature around
it. Can we find “less greedy” approaches? We can view variational
inference more generally as a family of approaches aiming to approximate the true posterior distribution by one that is closest (according
to some criterion) among a “simpler” class of distributions. To this
end, we need to fix a class of distributions and define suitable criteria,
which we can then optimize numerically. The key benefit is that we
can reduce the (generally intractable) problem of high-dimensional integration to the (often much more tractable) problem of optimization.

**Definition 5.17** (Variational family) **.** Let *P* be the class of all probability distributions. A *variational family* *Q ⊆P* is a class of distributions
such that each distribution *q* *∈Q* is characterized by unique variational parameters ***λ*** *∈* Λ.




Figure 5.8: An illustration of variational
inference in the space of distributions *P* .
The variational distribution *q* *[⋆]* *∈Q* is the
optimal approximation of the true posterior *p* .

104 probabilistic artificial intelligence



A common notion of distance between two distributions *q* and *p* is the
Kullback-Leibler divergence KL ( *q* *∥* *p* ) which we will define in the next
section. Using this notion of distance, we need to solve the following
optimization problem, [6] 6 Note that the order of *p* and *q* is reversed when compared to the defini
*q* *[⋆]* = . arg min KL ( *q* *∥* *p* ) = arg min KL ( *q* ***λ*** *∥* *p* ) . (5.42) tion of the KL-divergence. In the fol*q* *∈Q* ***λ*** *∈* Λ lowing section, we will first define the

KL-divergence and then (in section 5.5.1)
also explain why the order is reversed
here.


*q* *[⋆]* = . arg min KL ( *q* *∥* *p* ) = arg min KL ( *q* ***λ*** *∥* *p* ) . (5.42)
*q* *∈Q* ***λ*** *∈* Λ


*5.5* *Kullback-Leibler Divergence*


As mentioned, the Kullback-Leibler divergence is a (non-metric) measure of distance between distributions. It is defined as follows.

**Definition 5.19** (Kullback-Leibler divergence, KL-divergence) **.** Given
two distributions *p* and *q*, the *Kullback-Leibler divergence* (or *relative en-*
*tropy* ) of *q* with respect to *p*,

KL ( *p* *∥* *q* ) = . H [ *p* *∥* *q* ] *−* H [ *p* ] (5.43)

= **E** ***θ*** *∼* *p* [ S [ *q* ( ***θ*** )] *−* S [ *p* ( ***θ*** )]] (5.44)


= **E** ***θ*** *∼* *p*


log *[p]* [(] ***[θ]*** [)]
� *q* ( ***θ*** )


, (5.45)
�


measures how different *q* is from a reference distribution *p* .

In words, KL ( *p* *∥* *q* ) measures the *additional* expected surprise when observing samples from *p* that is due to assuming the (wrong) distribution *q* and which not inherent in the distribution *p* already. [7] Intuitively, 7 The KL-divergence only captures the

the KL-divergence measures the information loss when approximating additional expected surprise as the sur
prise inherent in *p* (as measured by H [ *p* ] )

*p* with *q* . is subtracted.

The KL-divergence has the following properties:

- KL ( *p* *∥* *q* ) *≥* 0 for any distributions *p* and *q*,

- KL ( *p* *∥* *q* ) = 0 if and only if *p* = *q* almost surely, and

- there exist distributions *p* and *q* such that KL ( *p* *∥* *q* ) *̸* = KL ( *q* *∥* *p* ) .

The KL-divergence can simply be understood as a shifted version of
cross-entropy, which is zero if we consider the divergence between two

identical distributions.

variational inference 105








106 probabilistic artificial intelligence





















variational inference 107

0.8


|Col1|q⋆ 2 q⋆ 1 p|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||


*−* 2 0 2 4 6

*x*

*x* 2

|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||−2 0 2||||
||||||
||||||
||||||
||||||
||||||



*−* 2 0 2

*x* 1






*5.5.1* *Forward and reverse KL-divergence*

KL ( *p* *∥* *q* ) is also called the *forward* (or *inclusive* ) KL-divergence. In
contrast, KL ( *q* *∥* *p* ) is called the *reverse* (or *exclusive* ) KL-divergence.
Figure 5.9 shows the approximations of a general Gaussian obtained
when *Q* is the family of diagonal (independent) Gaussians. Thereby,

*q* 1 *[⋆]* = . arg min KL ( *p* *∥* *q* ) *q* 2 *[⋆]* = . arg min KL ( *q* *∥* *p* )
*q* *∈Q* *q* *∈Q*

*q* 1 *[⋆]* [is the result when using the forward KL-divergence and] *[ q]* 2 *[⋆]* [is the re-]
sult when using reverse KL-divergence. It can be seen that the reverse
KL-divergence tends to greedily select the mode and underestimating the variance which, in this case, leads to an overconfident prediction. The forward KL-divergence, in contrast, is more conservative and
yields what one could consider the “desired” approximation.


Figure 5.9: Comparison of the **forward**
KL-divergence *q* 1 *[⋆]* [and the] **[ reverse]** [ KL-]
divergence *q* 2 *[⋆]* [when used to approxi-]
mate the **true posterior** *p* . The first plot
shows the PDFs in a one-dimensional
feature space where *p* is a mixture of
two univariate Gaussians. The second
plot shows contour lines of the PDFs in
a two-dimensional feature space where
the non-diagonal Gaussian *p* is approximated by diagonal Gaussians *q* 1 *[⋆]* [and] *[ q]* 2 *[⋆]* [.]
It can be seen that *q* 1 *[⋆]* [selects the vari-]
ance and *q* 2 *[⋆]* [selects the mode of] *[ p]* [. The]
approximation *q* 1 *[⋆]* [is more conservative]
than the (overconfident) approximation
*q* 2 *[⋆]* [.]


0.6

0.4

0.2

0.0

2

1

0

*−* 1

*−* 2

2

1

0

*−* 1

*−* 2


108 probabilistic artificial intelligence










Recall that in eq. (5.42) we use the reverse KL-divergence, which is typically used in practice. This is for computational reasons. Observe that
to approximate the KL-divergence KL ( *p* *∥* *q* ) using Monte Carlo sampling, we would need to obtain samples from *p* . But *p* is the intractable
posterior distribution which we were trying to approximate in the first
place. In principle, if the variational family *Q* is “rich enough”, minimizing the forward and reverse KL-divergences will yield the same

result.


variational inference 109




a sample mean and a sample variance
to compute the estimates of the first and
second moment.














vec [ ***A*** ] *∈* **R** *[n]* *[·]* *[m]*

to denote the row-by-row concatenation
of ***A*** yielding a vector of length *n* *·* *m* .

expanding the inner product and using
that tr ( *x* ) = *x* for all *x* *∈* **R**






cyclic permutations


***A***, ***B*** *∈* **R** *[n]* *[×]* *[n]*

110 probabilistic artificial intelligence














model parameters ***θ*** for estimating *y* *|*

rameters ***λ*** of a distribution over ***θ*** .


variational inference 111





(5.43)



*5.6* *Evidence Lower Bound*

The Evidence Lower Bound is a quantity maximization of which is
equivalent to minimizing the KL-divergence. This we will see later,
first let us define the evidence lower bound.

**Definition 5.31** (Evidence lower bound, ELBO) **.** The *evidence lower*
*bound* for data *D* = *{* ( ***x*** *i*, *y* *i* ) *}* *i* *[n]* = 1 [of a variational posterior] *[ q]* [ and the]
true posterior *p* ( *· |* ***x*** 1: *n*, *y* 1: *n* ) is defined as

*L* ( *q*, *p* ; *D* ) = . **E** ***θ*** *∼* *q* [ log *p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* )] + H [ *q* ] (5.61a)

= **E** ***θ*** *∼* *q* [ log *p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* ) *−* log *q* ( ***θ*** )] (5.61b) using the definition of entropy (5.29)

= **E** ***θ*** *∼* *q* [ log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) + log *p* ( ***θ*** ) *−* log *q* ( ***θ*** )] (5.61c) using the product rule (1.14)

= **E** ***θ*** *∼* *q* [ log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** )] *−* KL ( *q* *∥* *p* ( *·* )) (5.61d) using the definition of KL-divergence
(5.43)

*·*
where we denote by *p* ( ) the prior distribution.

Maximizing the evidence lower bound therefore corresponds to maximizing entropy and the joint likelihood of data and model. [12] In other 12 A joint likelihood (i.e., a prior *p* ( ***θ*** ) and
words, maximizing the data likelihood is traded with remaining close likelihood *p* ( ***y*** *|* ***θ*** ) ) is also called a *gener-*
*ative model* .
to the prior distribution.

112 probabilistic artificial intelligence

As its name suggests, the evidence lower bound is a (uniform) lower
bound to the log-evidence, log *p* ( *y* 1: *n* *|* ***x*** 1: *n* ) . To show this fact, we first

need to recall

Now, we are ready to prove that the evidence lower bound is a uniform [13] lower bound to the log-evidence, 13 That is, the bound holds for any variational distribution *q* (with full support).


log *p* ( *y* 1: *n* *|* ***x*** 1: *n* ) = log *p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* ) *d* ***θ*** by conditioning on ***θ*** using the sum rule
� (1.10)



*[|]* ***[ x]*** [1:] *[n]* [)]
= log *q* ( ***θ*** ) *[p]* [(] *[y]* [1:] *[n]* [,] ***[ θ]***
� ( ***θ*** )


*d* ***θ*** extending with *[q]* [(] ***[θ]*** [)] / *q* ( ***θ*** )
*q* ( ***θ*** )


interpreting integration as an
expectation over *q*


= log **E** ***θ*** *∼* *q*


*p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* )
� *q* ( ***θ*** )


*p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* )
� *q* ( ***θ*** )


�


*p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* )
log
� � *q* ( ***θ*** )


*≥* **E** ***θ*** *∼* *q*


*q* ( ***θ*** )


*≥* **E** ***θ*** *∼* *q* log *q* ( ***θ*** ) using Jensen’s inequality (concavity of the logarithm5.33) and

= **E** ***θ*** *∼* *q* [ log *p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* )] + *H* [ *q* ] using linearity of expectation (1.24) and
the definition of entropy (5.29)
= *L* ( *q*, *p* ; *D* ) . (5.62)


��


This indicates that maximizing the evidence lower bound is an adequate method of model selection which can be used instead of maximizing the evidence (marginal likelihood) directly (as was discussed
in section 4.4.2). Note that this inequality lower bounds the logarithm
of an integral by an expectation of a logarithm over some variational
distribution *q* . Hence, the ELBO is a family of lower bounds — one for
each variational distribution. Such inequalities are called variational
inequalities.

Next, let us see how the evidence lower bound can be used for learning
an approximate posterior distribution.


*q* ( ***θ*** )
log
� *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* )


using the definition of the
KL-divergence (5.43)


KL ( *q* *∥* *p* ( *· |* ***x*** 1: *n*, *y* 1: *n* )) = **E** ***θ*** *∼* *q*


�


log *[p]* [(] *[y]* [1:] *[n]* *[ |]* ***[ x]*** [1:] *[n]* [)] *[q]* [(] ***[θ]*** [)]
� *p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* )


�


= **E** ***θ*** *∼* *q*


*p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* )


= **E** ***θ*** *∼* *q* log *p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* ) using the definition of conditionalprobability (1.11)

= log *p* ( *y* 1: *n* *|* ***x*** 1: *n* ) using linearity of expectation (1.24


using linearity of expectation (1.24)


*−* **E** ***θ*** *∼* *q* [ log *p* ( *y* 1: *n*, ***θ*** *|* ***x*** 1: *n* )] *−* H [ *q* ]

= log *p* ( *y* 1: *n* *|* ***x*** 1: *n* ) *−* *L* ( *q*, *p* ; *D* ) . (5.63) using the definition of the ELBO (5.61a)

This gives the relationship

*L* ( *q*, *p* ; *D* ) = log *p* ( *y* 1: *n* *|* ***x*** 1: *n* ) *−* KL ( *q* *∥* *p* ( *· |* ***x*** 1: *n*, *y* 1: *n* )) . (5.64)
� �� �
const

Thus, maximizing the ELBO coincides with minimizing reverse-KL.
We have successfully recast the difficult problems of learning and inference as an optimization problem!

variational inference 113

Crucially, observe that if the true posterior *p* ( *· | D* ) is in the variational
family *Q*, then

arg max *L* ( *q*, *p* ; *D* ) = arg min KL ( *q* *∥* *p* ( *· | D* )) [a.s.] = *p* ( *· | D* ), (5.65) as min *q* *∈Q* KL ( *q* *∥* *p* ( *· | D* )) [a.s.] = 0
*q* *∈Q* *q* *∈Q*

implying that in this case, maximizing the ELBO recovers the true
posterior almost surely.




*−* H [ *q* ( ***θ*** )] (5.66) by negating eq. (5.62)






the two terms are commonly referred to
as “inaccuracy” and “complexity”,
mirroring our previous interpretation of
eq. (5.61d)


. using the derivation of eq. (5.63), the
“extrinsic” value is independent of the
variational distribution *q*, the
(5.69) “epistemic” value can be interpreted as
the approximation error.





scientist Karl Friston. See: “The free
ory?” (Friston)


114 probabilistic artificial intelligence


for an introduction to the “exploration


16 We explore this connection in remark 12.19.



simplicity.












using independence of the data

(5.70) substituting the logistic loss (5.14)


*5.6.1* *Gradient of Evidence Lower Bound*

The next problem is, of course, solving this optimization problem. In
stochastic gradient descent, we already know a tool which we can use.
However, SGD requires unbiased gradient estimates of the loss, which
is given as *ℓ* ( ***λ*** ; *D* ) = . *−* *L* ( *q* ***λ***, *p* ; *D* ) . Thus, we need to obtain gradient

variational inference 115

estimates of

***∇*** ***λ*** *L* ( *q* ***λ***, *p* ; *D* ) = ***∇*** ***λ*** **E** ***θ*** *∼* *q* ***λ*** [ log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** )] *−* ***∇*** ***λ*** KL ( *q* ***λ*** *∥* *p* ( *·* )) . using the definition of the ELBO (5.61d)
(5.71)

Typically, the KL-divergence (and its gradient) can be computed exactly for commonly used variational families. For example, we have
already seen a closed-form expression of the KL-divergence for Gaussians in eq. (5.50).







Obtaining the gradient of the evidence lower bound is more difficult.
This is because the expectation integrates over the measure *q* ***λ***, which
depends on the variational parameters ***λ*** . Thus, we cannot move the
gradient operator inside the expectation as we have done previously
in remark 1.24.

There are two main techniques which are used to rewrite the gradient
in such a way that Monte Carlo sampling becomes possible. The first is
to use *score gradients*, which we introduce in section 12.3.2. The second
is the so-called reparameterization trick.

**Theorem 5.35** (Reparameterization trick) **.** *Given a random variable* ***ϵ*** *∼* *ϕ*
*(which is independent of* ***λ*** *) and a differentiable and invertible function* ***g*** :
**R** *[d]* *→* **R** *[d]* *. We let* ***θ*** = [.] ***g*** ( ***ϵ*** ; ***λ*** ) *. Then,*

*q* ***λ*** ( ***θ*** ) = *ϕ* ( ***ϵ*** ) *· |* det ( ***D*** ***ϵ*** ***g*** ( ***ϵ*** ; ***λ*** )) *|* *[−]* [1], (5.73)

**E** ***θ*** *∼* *q* ***λ*** [ ***f*** ( ***θ*** )] = **E** ***ϵ*** *∼* *ϕ* [ ***f*** ( ***g*** ( ***ϵ*** ; ***λ*** ))] (5.74)

*for a “nice” function* ***f*** : **R** *[d]* *→* **R** *[e]* *.*

*Proof.* By the change of variables formula (1.52) and using ***ϵ*** = ***g*** *[−]* [1] ( ***θ*** ; ***λ*** ),

*q* ***λ*** ( ***θ*** ) = *ϕ* ( ***ϵ*** ) *·* det ***D*** ***θ*** ***g*** *[−]* [1] ( ***θ*** ; ***λ*** )
��� � ����

116 probabilistic artificial intelligence

= *ϕ* ( ***ϵ*** ) *·* det ( ***D*** ***ϵ*** ***g*** ( ***ϵ*** ; ***λ*** )) *[−]* [1] [���] by the inverse function theorem,
��� � �

***Dg*** *[−]* [1] ( ***y*** ) = ***Dg*** ( ***x*** ) *[−]* [1]

= *ϕ* ( ***ϵ*** ) *· |* det ( ***D*** ***ϵ*** ***g*** ( ***ϵ*** ; ***λ*** )) *|* *[−]* [1] . using det� ***A*** *[−]* [1] [�] = det ( ***A*** ) *[−]* [1]

Equation (5.74) is a direct consequence of the law of the unconscious
statistician (1.30).

Applying the reparameterization trick, we can use our analysis of
eq. (1.28) to swap the order of gradient and expectation,

***∇*** ***λ*** **E** ***θ*** *∼* *q* ***λ*** [ ***f*** ( ***θ*** )] = **E** ***ϵ*** *∼* *ϕ* [ ***∇*** ***λ*** ***f*** ( ***g*** ( ***ϵ*** ; ***λ*** ))] . (5.75)

If we can find a function ***g*** such that ***θ*** *∼* *q* ***λ*** for some sampling distribution *ϕ* independent of ***λ***, we can use the reparameterization trick
to simplify our gradient. We call a distribution *q* ***λ*** which admits reparameterization *reparameterizable* .



the change of variables formula) (5.73)






variational inference 117



In the following, we write ***C*** = . ***Σ*** 1 / 2 . Let us now derive the gradient
estimate for the evidence lower bound assuming the Gaussian variational approximation from example 5.36. This approach extends to
any reparameterizable distribution.

***∇*** ***λ*** **E** ***θ*** *∼* *q* ***λ*** [ log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** )]

= ***∇*** ***C***, ***µ*** **E** ***ϵ*** *∼N* ( **0**, ***I*** ) � log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) *|* ***θ*** = ***Cϵ*** + ***µ*** � (5.80) using the reparameterization trick (5.74)


using independence of the data and
extending with *[n]* / *n*


�


= *n* *·* ***∇*** ***C***, ***µ*** **E** ***ϵ*** *∼N* ( **0**, ***I*** )


1

*n*
�


*n*
###### ∑ log p ( y i | x i, θ ) | θ = Cϵ + µ

*i* = 1


= *n* *·* ***∇*** ***C***, ***µ*** **E** ***ϵ*** *∼N* ( **0**, ***I*** ) **E** *i* *∼* Unif ([ *n* ]) � log *p* ( *y* *i* *|* ***x*** *i*, ***θ*** ) *|* ***θ*** = ***Cϵ*** + ***µ*** � interpreting the sum as an expectation

= *n* *·* **E** ***ϵ*** *∼N* ( **0**, ***I*** ) **E** *i* *∼* Unif ([ *n* ]) � ***∇*** ***C***, ***µ*** log *p* ( *y* *i* *|* ***x*** *i*, ***θ*** ) *|* ***θ*** = ***Cϵ*** + ***µ*** � (5.81) using eq. (1.28)


*≈* *n* *·* [1]

*m*


*m*
###### ∑ ∇ C, µ log p ( y i ( j ) | x i ( j ), θ ) �� θ = Cϵ [(] [j] [)] + µ (5.82) using Monte Carlo sampling

*j* = 1


where ***ϵ*** [(] *[j]* [)] [ iid] *∼N* ( **0**, ***I*** ) and *i* [(] *[j]* [)] [ iid] *∼* Unif ([ *n* ]) . This yields an unbiased
gradient estimate, which we can use with stochastic gradient descent

to maximize the evidence lower bound.

The procedure of approximating the true posterior using a variational
posterior by maximizing the evidence lower bound using stochastic
optimization is also called *black box stochastic variational inference* . The
only requirement is that we can obtain unbiased gradient estimates
from the evidence lower bound (and the likelihood). If we use the variational family of diagonal Gaussians, we only require twice as many
parameters as other inference techniques like MAP estimation. The
performance can be improved by using natural gradients and variance
reduction techniques for the gradient estimates such as control vari
ates.

118 probabilistic artificial intelligence


### *6* *Markov Chain Monte Carlo Methods*

Variational inference approximates the entire posterior distribution.
However, note that the key challenge in Bayesian inference is not learning the posterior distribution, but using the posterior distribution for
predictions,

*p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ) = *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ) *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) *d* ***θ*** . (6.1)
�

This integral can be interpreted as an expectation over the posterior
distribution,

= **E** ***θ*** *∼* *p* ( *·|* ***x*** 1: *n*, *y* 1: *n* ) [ *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** )] . (6.2)

Observe that the likelihood *f* ( ***θ*** ) = . *p* ( *y* *⋆* *|* ***x*** *⋆*, ***θ*** ) is easy to evaluate. The
difficulty lies in sampling from the posterior distribution. Assuming
we can obtain independent samples from the posterior distribution,
we can use Monte Carlo sampling to obtain an unbiased estimate of
the expectation,


*≈* [1]

*m*


*m*
###### ∑ f ( θ [(] [i] [)] ) (6.3)

*i* = 1


for independent ***θ*** [(] *[i]* [)] [ iid] *∼* *p* ( *· |* ***x*** 1: *n*, *y* 1: *n* ) . The law of large numbers (1.83)
and Hoeffding’s inequality (1.87) imply that this estimator is consistent
and sharply concentrated. [1] 1 For more details, refer to section 1.4.2.

Obtaining samples of the posterior distribution is therefore sufficient
to perform approximate inference. Recall that the difficulty of computing the posterior *p* exactly, was in finding the normalizing constant

*Z*,

*p* ( *x* ) = [1] (6.4)

*Z* *[q]* [(] *[x]* [)] [.]

The joint likelihood *q* is typically easy to obtain. Note that *q* ( *x* ) is proportional to the probability density associated with *x*, but *q* does not

120 probabilistic artificial intelligence

integrate to 1. Such functions are also called a *finite measure* . Without
normalizing *q*, we cannot directly sample from it.

Even if we were able to approximate *Z*, this generally does not yield
an efficient sampling method. For example, the inverse transform
sampling discussed in remark 1.14 requires an (approximate) quantile function. Computing the quantile function given an arbitrary PDF
requires solving integrals over the domain of the PDF which is what
we were trying to avoid in the first place.

The key idea of Markov chain Monte Carlo methods is to construct
a Markov chain, which is efficient to simulate and has the stationary
distribution *p* .

*6.1* *Markov Chains*

To begin with, let us revisit the fundamental theory behind Markov

chains.

**Definition 6.1** (Markov chain) **.** A *(finite and discrete-time) Markov chain*
over the state space

*S* = . *{* 0, . . ., *n* *−* 1 *}* (6.5)

is a stochastic process [2] ( *X* *t* ) *t* *∈* **N** 0 valued in *S* such that the *Markov prop-* 2 A *stochastic process* is a sequence of random variables.
*erty*,


*X* *t* + 1 *⊥* *X* 0: *t* *−* 1 *|* *X* *t*, (6.6)



is satisfied. Figure 6.1: Directed graphical model of

a Markov chain. The random variable
Intuitively, the Markov property states that future behavior is indepen- *X* *t* + 1 is conditionally independent of the
random variables *X* 0: *t* *−* 1 given *X* *t* .
dent of past states given the present state.

We restrict our attention to *time-homogeneous* Markov chains, [3] which 3 That is, the transition probabilities do
not change over time.
can be characterized by a *transition function*,

*p* ( *x* *[′]* *|* *x* ) = . **P** � *X* *t* + 1 = *x* *[′]* *|* *X* *t* = *x* �. (6.7)

markov chain monte carlo methods 121

As the state space is finite, we can describe the transition function by
the *transition matrix*,




 *∈* **R** *n* *×* *n* . (6.8)


***P*** = .







*p* ( *x* 1 *|* *x* 1 ) *· · ·* *p* ( *x* *n* *|* *x* 1 )

... ... ...

*p* ( *x* 1 *|* *x* *n* ) *· · ·* *p* ( *x* *n* *|* *x* *n* )


Note that each row of ***P*** must always sum to 1. Such matrices are also

called *stochastic* .

The *transition graph* of a Markov chain is a directed graph consisting of
vertices *S* and weighted edges represented by the adjacency matrix ***P*** .

The current state of the Markov chain at time *t* is denoted by the probability distribution *q* *t* over states *S*, that is, *X* *t* *∼* *q* *t* . In the finite setting, *q* *t* is a PMF, which is often written explicitly as the row vector
***q*** *t* *∈* **R** [1] *[×|]* *[S]* *[|]* . The initial state (or prior) of the Markov chain is given as
*X* 0 *∼* *q* 0 .




It is implied directly that we can write the state of the Markov chain
at time *t* + *k* as

***q*** *t* + *k* = ***q*** *t* ***P*** *[k]* . (6.10)




In the analysis of Markov chains, there are two main concepts of interest: stationarity and convergence. We begin by introducing stationarity.

122 probabilistic artificial intelligence

*6.1.1* *Stationarity*

**Definition 6.5** (Stationary distribution) **.** A distribution *π* is *stationary*
with respect to the transition function *p* iff
###### π ( x ) = ∑ p ( x | x [′] ) π ( x [′] ) (6.11)

*x* *[′]* *∈* *S*

holds for all *x* *∈* *S* . It follows from eq. (6.9) that equivalently, *π* is
stationary w.r.t. a transition matrix ***P*** iff

***π*** = ***πP*** . (6.12)

After entering a stationary distribution *π*, a Markov chain will always
remain in the stationary distribution. In particular, suppose that *X* *t* is
distributed according to *π*, then for all *k* *≥* 0, *X* *t* + *k* *∼* *π* .



*6.1.2* *Convergence*

Let us now consider Markov chains with a unique stationary distribution. [4] A natural next question is whether this Markov chain converges 4 Observe that the stationary distribu
to its stationary distribution. We say that a Markov chain converges to tion of an irreducible Markov chain must

have full support, that is, assign positive

its stationary distribution iff we have probability to every state.

lim (6.14)
*t* *→* ∞ *[q]* *[t]* [ =] *[ π]* [,]

irrespectively of the initial distribution *q* 0 .


markov chain monte carlo methods 123



This additional property leads to the concept of ergodicity.

**Definition 6.8** (Ergodicity) **.** A Markov chain is *ergodic* iff there exists a
*t* *∈* **N** 0 such that for any *x*, *x* *[′]* *∈* *S* we have

*p* [(] *[t]* [)] ( *x* *[′]* *|* *x* ) *>* 0, (6.16)

whereby *p* [(] *[t]* [)] ( *x* *[′]* *|* *x* ) is the probability to reach *x* *[′]* from *x* in exactly *t*
steps. Equivalent conditions are

1. that there exists some *t* *∈* **N** 0 such that all entries of ***P*** *[t]* are strictly
positive; and
2. that it is irreducible and aperiodic.


1 / 2


1 1 2 1




Figure 6.2: Transition graphs of Markov
chains: (1) is not ergodic as its transition diagram is not strongly connected;
(2) is not ergodic for the same reason;
(3) is periodic and therefore not ergodic;
(4) is ergodic with stationary distribution *π* ( 1 ) = [2] / 3, *π* ( 2 ) = [1] / 3 .





124 probabilistic artificial intelligence







(6.18) using (6.12)


**Fact 6.10** (Fundamental theorem of ergodic Markov chains) **.** *An ergodic*
*Markov chain has a unique stationary distribution π (with full support) and*

lim (6.19)
*t* *→* ∞ *[q]* *[t]* [ =] *[ π]*

*irrespectively of the initial distribution q* 0 *.*

*Proof.* Refer to theorem 4.9 of “Markov chains and mixing times” (Levin
and Peres) for a proof.

This naturally suggests constructing an ergodic Markov chain such
that its stationary distribution coincides with the posterior distribution. If we then sample “sufficiently long”, *X* *t* is drawn from a distribution that is “very close” to the posterior distribution.


markov chain monte carlo methods 125




126 probabilistic artificial intelligence

*6.1.3* *Detailed Balance Equation*

How can we confirm that the stationary distribution of a Markov chain
coincides with the posterior distribution? The detailed balance equation yields a very simple method.

**Definition 6.15** (Detailed balance equation / reversibility) **.** A Markov
chain satisfies the *detailed balance equation* with respect to a distribution

*π* iff

*π* ( *x* ) *p* ( *x* *[′]* *|* *x* ) = *π* ( *x* *[′]* ) *p* ( *x* *|* *x* *[′]* ) (6.24)

holds for any *x*, *x* *[′]* *∈* *S* . A Markov chain that satisfies the detailed
balance equation with respect to *π* is called *reversible* with respect to

*π* .

**Lemma 6.16.** *Given a finite Markov chain, if the Markov chain is reversible*
*with respect to π then π is a stationary distribution.* [5] 5 Note that reversibility of *π* is only a sufficient condition for stationarity of *π*, it
*Proof.* Let *π* = . *q* *t* . We have, is not necessary! In particular, there areirreversible ergodic Markov chains.
###### q t + 1 ( x ) = ∑ p ( x | x [′] ) q t ( x [′] ) using the Markov property (6.6)

*x* *[′]* *∈* *S*
###### = ∑ p ( x | x [′] ) π ( x [′] )

*x* *[′]* *∈* *S*
###### = ∑ p ( x [′] | x ) π ( x ) using the detailed balance equation

*x* *[′]* *∈* *S* (6.24)
###### = π ( x ) ∑ p ( x [′] | x )

*x* *[′]* *∈* *S*

= *π* ( *x* ) . using that ∑ *x* *′* *∈* *S* *p* ( *x* *[′]* *|* *x* ) = 1

That is, if we can show that the detailed balance equation (6.24) holds
for some distribution *q*, then we know that *q* is the stationary distribu
tion of the Markov chain.

Next, reconsider our posterior distribution *p* ( *x* ) = *Z* [1] *[q]* [(] *[x]* [)] [ from eq. (][6][.][4][).]

If we substitute the posterior for *π* in the detailed balance equation, we

obtain


or equivalently,


1 [1] (6.25)
*Z* *[q]* [(] *[x]* [)] *[p]* [(] *[x]* *[′]* *[ |]* *[ x]* [) =] *Z* *[q]* [(] *[x]* *[′]* [)] *[p]* [(] *[x]* *[ |]* *[ x]* *[′]* [)] [,]

*q* ( *x* ) *p* ( *x* *[′]* *|* *x* ) = *q* ( *x* *[′]* ) *p* ( *x* *|* *x* *[′]* ) . (6.26)


In words, we do not need to know the true posterior *p* to check that
the stationary distribution of our Markov chain coincides with *p*, it
suffices to know the finite measure *q* !

markov chain monte carlo methods 127

*6.1.4* *Ergodic Theorem*

If we now suppose that we can construct a Markov chain whose stationary distribution coincides with the posterior distribution — we
will see later that this is possible — it is not apparent that this allows
us to estimate expectations over the posterior distribution. Note that
although constructing such a Markov chain allows us to obtain samples from the posterior distribution, they are *not* independent. In fact,
due to the structure of a Markov chain, by design, they are strongly
dependent. Thus, the law of large numbers and Hoeffding’s inequality
do not apply. By itself, it is not even clear that an estimator relying on
samples from a single Markov chain will be unbiased.

Theoretically, we could simulate many Markov chains separately and
obtain one sample from each of them. This, however, is extremely
inefficient.

It turns out that there is a way to generalize the (strong) law of large

numbers to Markov chains.

**Theorem 6.17** (Ergodic theorem) **.** *Given an ergodic Markov chain* ( *X* *t* ) *t* *∈* **N** 0
*over a finite state space S with stationary distribution π and a function*
*f* : *S* *→* **R** *,*


1

*n*


*n*
###### ∑ f ( x i ) [a.s.] → ∑ π ( x ) f ( x ) = E x ∼ π [ f ( x )] (6.27)

*i* = 1 *x* *∈* *S*


*as n* *→* ∞ *where x* *i* *∼* *X* *i* *|* *x* *i* *−* 1 *.*

*Proof.* See appendix C of “Markov chains and mixing times” (Levin
and Peres) for a proof.

This result is the fundamental reason for why Markov chain Monte
Carlo methods are possible. There are analogous results for continu
ous domains.

Note, however, that the ergodic theorem only tells us that simulating
a single Markov chain yields an unbiased estimator. It does not tell
us anything about the rate of convergence and variance of such an
estimator. The convergence rate depends on the mixing time of the
Markov chain, which is difficult to establish in general.

In practice, one observes that Markov chain Monte Carlo methods have
a so-called *“burn-in” time* during which the distribution of the Markov
chain does not yet approximate the posterior distribution well. Typically, the first *t* 0 samples are therefore discarded,


1

*p*

0

*t* 0

*t*

Figure 6.3: Illustration of the “burn-in”
time *t* 0 of a Markov chain approximating the posterior *p* ( *y* *[⋆]* = 1 *|* ***X***, ***y*** ) of
Bayesian logistic regression. The true
posterior *p* is shown in gray. The distribution of the Markov chain at time *t* is

shown in red.


1
**E** [ *f* ( *X* )] *≈*
*T* *−* *t* 0


*T*
###### ∑ f ( X t ) . (6.28)

*t* = *t* 0 + 1

128 probabilistic artificial intelligence

It is not clear in general how *T* and *t* 0 should be chosen such that the
estimator is unbiased, rather they have to be tuned.

Another widely used heuristic is to first find the mode of the posterior
distribution and then start the Markov chain at that point. This tends
to reduce the rate of convergence drastically, as the Markov chain does
not have to “walk to the location in the state space where most probability mass will be located”.

*6.2* *Elementary Sampling Methods*

We will now examine methods for constructing and sampling from a
Markov chain with the goal of approximating samples from the posterior distribution *p* . Note that in this setting the state space of the
Markov chain is **R** *[n]* and a single state at time *t* is described by the
random vector **X** = [ . *X* 1, . . ., *X* *n* ] .

*6.2.1* *Metropolis-Hastings Algorithm*

Suppose we are given a *proposal distribution r* ( ***x*** *[′]* *|* ***x*** ) which, given
we are in state ***x***, proposes a new state ***x*** *[′]* . Metropolis and Hastings
showed that using the *acceptance distribution* Bern ( *α* ( ***x*** *[′]* *|* ***x*** )) where


*α* ( ***x*** *[′]* *|* ***x*** ) = . min 1, *[p]* [(] ***[x]*** *[′]* [)] *[r]* [(] ***[x]*** *[|]* ***[ x]*** *[′]* [)]
� *p* ( ***x*** ) *r* ( ***x*** *[′]* *|* ***x*** )

*[|]* ***[ x]*** *[′]* [)]
= min 1, *[q]* [(] ***[x]*** *[′]* [)] *[r]* [(] ***[x]***
� *q* ( ***x*** ) *r* ( ***x*** *[′]* *|* ***x*** )


(6.29)
�


�


(6.30)


similarly to the detailed balance
equation, the normalizing constant *Z*
cancels


to decide whether to follow the proposal yields a Markov chain with
stationary distribution *p* ( ***x*** ) = *Z* [1] *[q]* [(] ***[x]*** [)] [.]


**Al** **g** **orithm 6.18:** Metro p olis-Hastin g s al g orithm

**1** initialize ***x*** *∈* **R** *[n]*

**2** `for` *t* = 1 `to` *T* `do`

**3** sample ***x*** *[′]* *∼* *r* ( ***x*** *[′]* *|* ***x*** )

**4** sample *u* *∼* Unif ([ 0, 1 ])

**5** `if` *u* *≤* *α* ( ***x*** *[′]* *|* ***x*** ) `then` update ***x*** *←* ***x*** *[′]*

**6** `else` update ***x*** *←* ***x***

**Theorem 6.19** (Metropolis-Hastings theorem) **.** *Given an arbitrary pro-*
*posal distribution r, the stationary distribution of the Markov chain simulated*
*by the Metropolis-Hastings algorithm is p* ( ***x*** ) = *Z* [1] *[q]* [(] ***[x]*** [)] *[.]*

markov chain monte carlo methods 129

*Proof.* First, let us define the transition probabilities of the Markov
chain. The probability of transitioning from a state ***x*** to a state ***x*** *[′]* is
given by *r* ( ***x*** *[′]* *|* ***x*** ) *α* ( ***x*** *[′]* *|* ***x*** ) if ***x*** *̸* = ***x*** *[′]* and the probability of proposing to
remain in state ***x***, *r* ( ***x*** *|* ***x*** ), plus the probability of denying the proposal,

otherwise.

*̸*


*r* ( ***x*** *|* ***x*** ) + ∑ ***x*** *′′* *̸* = ***x*** *r* ( ***x*** *[′′]* *|* ***x*** )( 1 *−* *α* ( ***x*** *[′′]* *|* ***x*** )) otherwise.


*p* ( ***x*** *[′]* *|* ***x*** ) =

*̸*





*̸*


*r* ( ***x*** *[′]* *|* ***x*** ) *α* ( ***x*** *[′]* *|* ***x*** ) if ***x*** *̸* = ***x*** *[′]*

 *r* ( ***x*** *|* ***x*** ) + ∑ ***x*** *′′* *̸* = ***x*** *r* ( ***x*** *[′′]* *|* ***x*** )( 1 *−* *α* ( ***x*** *[′′]* *|* ***x*** ))


*̸*

(6.31)

We will show that the stationary distribution is *p* by showing that
*p* satisfies the detailed balance equation (6.24). Let us fix arbitrary
states ***x*** and ***x*** *[′]* . First, observe that if ***x*** = ***x*** *[′]*, then the detailed balance

equation is trivially satisfied. Without loss of generality we assume

*α* ( ***x*** *|* ***x*** *[′]* ) = 1, *α* ( ***x*** *[′]* *|* ***x*** ) = *[q]* [(] ***[x]*** *[′]* [)] *[r]* [(] ***[x]*** *[|]* ***[ x]*** *[′]* [)]

*q* ( ***x*** ) *r* ( ***x*** *[′]* *|* ***x*** ) [.]

For ***x*** *̸* = ***x*** *[′]*, we then have,

*p* ( ***x*** ) *·* *p* ( ***x*** *[′]* *|* ***x*** ) = [1] using the definition of the distribution *p*

*Z* *[q]* [(] ***[x]*** [)] *[p]* [(] ***[x]*** *[′]* *[ |]* ***[ x]*** [)]

= [1] using the transition probabilities of the

*Z* *[q]* [(] ***[x]*** [)] *[r]* [(] ***[x]*** *[′]* *[ |]* ***[ x]*** [)] *[α]* [(] ***[x]*** *[′]* *[ |]* ***[ x]*** [)]
Markov chain


*̸*

[1] *[|]* ***[ x]*** *[′]* [)]

*Z* *[q]* [(] ***[x]*** [)] *[r]* [(] ***[x]*** *[′]* *[ |]* ***[ x]*** [)] *[q]* [(] ***[x]*** ***x*** *[′]* [)] *r* *[r]* [(] ***x*** ***[x]*** *[′]* ***x***


*̸*

= [1]


*̸*

using the definition of the acceptance
*q* ( ***x*** ) *r* ( ***x*** *[′]* *|* ***x*** ) distribution *α*


*̸*

= [1]

*Z* *[q]* [(] ***[x]*** *[′]* [)] *[r]* [(] ***[x]*** *[ |]* ***[ x]*** *[′]* [)]


*̸*

= [1]


*̸*

= using the definition of the acceptance

*Z* *[q]* [(] ***[x]*** *[′]* [)] *[r]* [(] ***[x]*** *[ |]* ***[ x]*** *[′]* [)] *[α]* [(] ***[x]*** *[ |]* ***[ x]*** *[′]* [)]
distribution *α*
= [1] ***[x]*** *[′]* ***[x]*** ***[ x]*** *[′]*


*̸*

= using the transition probabilities of the

*Z* *[q]* [(] ***[x]*** *[′]* [)] *[p]* [(] ***[x]*** *[ |]* ***[ x]*** *[′]* [)]
Markov chain
= *p* ( ***x*** *[′]* ) *·* *p* ( ***x*** *|* ***x*** *[′]* ) . using the definition of the distribution *p*


*̸*

Note that by the fundamental theorem of ergodic Markov chains (6.19),
for convergence to the stationary distribution, it is sufficient for the
Markov chain to be ergodic. Ergodicity follows immediately when
the transition probabilities *p* ( *· |* ***x*** ) have full support. For example, if
the proposal distribution *r* ( *· |* ***x*** ) has full support, the full support of
*p* ( *· |* ***x*** ) follows immediately from eq. (6.31). The rate of convergence
of Metropolis-Hastings depends strongly on the choice of the proposal

distribution.

*6.2.2* *Gibbs Sampling*

A popular example of a Metropolis-Hastings algorithm is Gibbs sampling as presented in alg. 6.20.

130 probabilistic artificial intelligence

**Al** **g** **orithm 6.20:** Gibbs sam p lin g

**1** initialize ***x*** = [ *x* 1, . . ., *x* *n* ] *∈* **R** *[n]*

**2** `for` *t* = 1 `to` *T* `do`

**3** pick a variable *i* uniformly at random from *{* 1, . . ., *n* *}*

**4** set ***x*** *−* *i* = [ . *x* 1, . . ., *x* *i* *−* 1, *x* *i* + 1, . . ., *x* *n* ]

**5** update *x* *i* by sampling according to the posterior distribution
*p* ( *x* *i* *|* ***x*** *−* *i* )

Intuitively, by re-sampling single coordinates according to the posterior distribution given the other coordinates, Gibbs sampling finds
states that are successively “more” likely.

**Theorem 6.21** (Gibbs sampling as Metropolis-Hastings) **.** *Gibbs sam-*
*pling is a Metropolis-Hastings algorithm with proposal distribution*


(6.32)
0 *otherwise*


*r* ( ***x*** *[′]* *|* ***x*** ) = .






0 *p* ( *x* *i* *[′]* *[|]* ***[ x]*** *[′−]* *[i]* [)] *otherwise* ***x*** *[′]* *differs from* ***x*** *only in entry i*




*and acceptance distribution α* ( ***x*** *[′]* *|* ***x*** ) = . 1 *.*

*Proof.* We show that *α* ( ***x*** *[′]* *|* ***x*** ) = 1 follows from the definition of an
acceptance distribution in Metropolis-Hastings (6.30) and the choice
of proposal distribution (6.32).

By (6.30),


*α* ( ***x*** *[′]* *|* ***x*** ) = min 1, *[p]* [(] ***[x]*** *[′]* [)] *[r]* [(] ***[x]*** *[|]* ***[ x]*** *[′]* [)]
� *p* ( ***x*** ) *r* ( ***x*** *[′]* *|* ***x*** )


�


Note that *p* ( ***x*** ) = *p* ( *x* *i*, ***x*** *−* *i* ) = *p* ( *x* *i* *|* ***x*** *−* *i* ) *p* ( ***x*** *−* *i* ) using the product rule
(1.14). Therefore,


= min 1, *[p]* [(] *[x]* *i* *[′]* *[|]* ***[ x]*** *[′−]* *[i]* [)] *[p]* [(] ***[x]*** *[′−]* *[i]* [)] *[r]* [(] ***[x]*** *[|]* ***[ x]*** *[′]* [)]
� *p* ( *x* *i* *|* ***x*** *−* *i* ) *p* ( ***x*** *−* *i* ) *r* ( ***x*** *[′]* *|* ***x*** )


�


= min 1, *[p]* [(] *[x]* *i* *[′]* *[|]* ***[ x]*** *[′−]* *[i]* [)] *[p]* [(] ***[x]*** *[′−]* *[i]* [)] *[p]* [(] *[x]* *[i]* *[|]* ***[ x]*** *[−]* *[i]* [)]
� *p* ( *x* *i* *|* ***x*** *−* *i* ) *p* ( ***x*** *−* *i* ) *p* ( *x* *i* *[′]* *[|]* ***[ x]*** *[′−]* *[i]* [)]


�


using the proposal distribution (6.32)


= min 1, *[p]* [(] ***[x]*** *[′−]* *[i]* [)]
� *p* ( ***x*** *−* *i* )


�


= 1. using that *p* ( ***x*** *[′]* *−* *i* ) = *p* ( ***x*** *−* *i* )

**Corollary 6.22** (Convergence of Gibbs sampling) **.** *As Gibbs sampling*
*is a specific example of an MH-algorithm, the stationary distribution of the*
*simulated Markov chain is p* ( ***x*** ) *.*

markov chain monte carlo methods 131

Note that for the proposals of Gibbs sampling, we have

*p* ( *x* *i* *|* ***x*** *−* *i* ) = *p* ( *x* *i*, ***x*** *−* *i* ) = *q* ( *x* *i*, ***x*** *−* *i* ) . (6.33) using the definition of condition
� *p* ( *x* *i*, ***x*** *−* *i* ) *dx* *i* � *q* ( *x* *i*, ***x*** *−* *i* ) *dx* *i* probability (1.11) and the sum rule
(1.10)

Under many models, this probability can be efficiently evaluated due
to the conditioning on the remaining coordinates ***x*** *−* *i* . If *X* *i* has finite
support, the normalizer can be computed exactly. Even if *X* *i* were to
be a continuous random variable, the normalizer can be approximated
using numerical quadrature methods for one-dimensional integrals.







*6.3* *Langevin Dynamics*

*6.3.1* *Gibbs Distributions*

Gibbs distributions are a special class of distributions that are widely
used in machine learning.

**Definition 6.24** (Gibbs distribution) **.** A *Gibbs distribution* (also called a
*Boltzmann distribution* ) is a continuous distribution *p* whose PDF is of

the form

*p* ( ***x*** ) = *Z* [1] [exp] [(] *[−]* *[f]* [ (] ***[x]*** [))] [.] (6.34)

*f* : **R** *[n]* *→* **R** is also called an *energy function* . [6] 6 Note that an energy function can be interpreted as a loss function.
A useful property is that Gibbs distributions always have full support. [7] 7 This can easily be seen as exp ( *·* ) *>* 0.
When the energy function *f* is convex, its Gibbs distribution is called
*log-concave* .

132 probabilistic artificial intelligence

It is often easier to reason about “energies” rather than probabilities
as they are neither restricted to be non-negative nor do they have to
integrate to 1. We remark that energy lives in the space of surprise,
that is, *f* ( ***x*** ) corresponds to the unnormalized surprise S [ *p* ( ***x*** )] about
observing ***x*** *∼* *p* .





Observe that the posterior distribution can always be interpreted as a
Gibbs distribution as long as prior and likelihood functions are posi
tive,

*p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) = [1] using Bayes’ rule (1.58)

*Z* *[p]* [(] ***[θ]*** [)] *[p]* [(] *[y]* [1:] *[n]* *[ |]* ***[ x]*** [1:] *[n]* [,] ***[ θ]*** [)]

= *Z* [1] [exp] [(] *[−]* [[] *[−]* [log] *[ p]* [(] ***[θ]*** [)] *[ −]* [log] *[ p]* [(] *[y]* [1:] *[n]* *[ |]* ***[ x]*** [1:] *[n]* [,] ***[ θ]*** [)])] [.] (6.35)

Thus, defining the energy function

*f* ( ***θ*** ) = . *−* log *p* ( ***θ*** ) *−* log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) (6.36)

*n*
###### = − log p ( θ ) − ∑ log p ( y i | x i, θ ), (6.37)

*i* = 1

yields

*p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) = *Z* [1] [exp] [(] *[−]* *[f]* [ (] ***[θ]*** [))] [.] (6.38)

Note that *f* coincides with the loss function used for MAP estimation
(1.95). For a uniform prior, the regularization term vanishes and the
energy reduces to the negative log-likelihood *ℓ* nll ( ***θ*** ; *D* ) (i.e., the loss
function of maximum likelihood estimation (1.90)).

Using that the posterior is a Gibbs distribution, we can rewrite the
acceptance distribution of Metropolis-Hastings,

*α* ( ***x*** *[′]* *|* ***x*** ) = min 1, *[r]* [(] ***[x]*** *[|]* ***[ x]*** *[′]* [)] . (6.39) this is obtained by substituting the PDF
� *r* ( ***x*** *[′]* *|* ***x*** ) [exp] [(] *[ f]* [ (] ***[x]*** [)] *[ −]* *[f]* [ (] ***[x]*** *[′]* [))] � of a Gibbs distribution for the posterior

markov chain monte carlo methods 133









134 probabilistic artificial intelligence







*f* ( *θ* )


*6.3.2* *Stochastic Gradient Langevin Dynamics*

Until now, we have looked at Metropolis-Hastings algorithms with
proposal distributions that do not explicitly take into account the curvature of the energy function around the current state. Langevin dynamics adapts the Gaussian proposals of the Metropolis-Hastings algorithm we have seen in example 6.28 to search the state space in an
“informed” direction. The simple idea is to bias the sampling towards
states with lower energy, thereby making it more likely that a proposal
is accepted.

A natural idea is to shift the proposal distribution perpendicularly to
the gradient of the energy function. This yields the following proposal
distribution,

*r* ( ***x*** *[′]* *|* ***x*** ) = *N* ( ***x*** *[′]* ; ***x*** *−* *η* *t* ***∇*** *f* ( ***x*** ), 2 *η* *t* ***I*** ) . (6.44)

The resulting variant of Metropolis-Hastings is known as the *Metropolis*
*adjusted Langevin algorithm* (MALA), *Langevin Monte Carlo* (LMC), or
simply *Langevin dynamics* . It can be shown that, as *η* *t* *→* 0, we have
for the acceptance probability *α* ( ***x*** *[′]* *|* ***x*** ) *→* 1 using that the acceptance
probability is 1 if ***x*** *[′]* = ***x*** . Hence, the Metropolis-Hastings acceptance
step can be omitted once the rejection probability becomes negligible.


*θ*

Figure 6.4: Metropolis-Hastings and
Langevin dynamics minimize the energy function *f* ( *θ* ) shown in blue. Suppose we start at the black dot *θ* 0, then
the black and red arrows denote possible subsequent samples. MetropolisHastings uses an “uninformed” search
direction, whereas Langevin dynamics
uses the gradient of *f* ( *θ* ) to make “more
promising” proposals. The random proposals help get past local optima.

markov chain monte carlo methods 135

For log-concave distributions, the mixing time of the underlying Markov
chain can be shown to be polynomial in the dimension *n* . [9] Note, 9 Nawaf Bou-Rabee and Martin Hairer.

however, that computing the gradient of the energy function, which Nonasymptotic mixing of the mala algo
rithm. *IMA Journal of Numerical Analysis*,

corresponds to computing exact gradients of the log-prior and log- 33(1):80–110, 2013
likelihood, in every step can be expensive.

The proposal step can be made more efficient by approximating the
gradient with an unbiased gradient estimate, analogously to stochastic
gradient descent. Recall that we want to produce approximate samples

of


�


***θ*** *∼* *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) = *Z* [1] [exp]


�


*n*
###### log p ( θ ) + ∑ log p ( y i | x i, θ )

*i* = 1


(6.45)


and observe that for a general distribution *p* (which is not necessarily a
Gibbs distribution), the proposal ***θ*** [˜] of MALA/LMC can be equivalently

formulated as


�


˜
***θ*** *←* ***θ*** + *η* *t*


�


*n*
###### ∇ log p ( θ ) + ∑ ∇ log p ( y i | x i, θ )

*i* = 1


+ *ϵ* (6.46)


where *ϵ* *∼N* ( **0**, 2 *η* *t* ***I*** ) .

*Stochastic gradient Langevin dynamics* (SGLD) (see alg. 6.29) combines
ideas from stochastic gradient descent and Metropolis-Hastings, [10] 10 Max Welling and Yee W Teh. Bayesian
learning via stochastic gradient langevin
dynamics. In *Proceedings of the 28th in-*
*ternational conference on machine learning*
**Al** **g** **orithm 6.29:** Stochastic g radient Lan g evin d y namics, SGLD *(ICML-11)*, pages 681–688. Citeseer, 2011

**1** initialize ***θ***

**2** `for` *t* = 1 `to` *T* `do`

**3** sample *i* 1, . . ., *i* *m* *∼* Unif ( *{* 1, . . ., *n* *}* ) independently

**4** sample *ϵ* *∼N* ( **0**, 2 *η* *t* ***I*** )

**5** ***θ*** *←* ***θ*** + *η* *t* � ***∇*** log *p* ( ***θ*** ) + *m* *[n]* [∑] *[m]* *j* = 1 ***[∇]*** [log] *[ p]* [(] *[y]* *[i]* *j* *[|]* ***[ x]*** *[i]* *j* [,] ***[ θ]*** [)] � + *ϵ* `//` (6.47)

Observe that SGLD (6.47) differs from LMC (6.46) by using a samplingbased approximation of the gradient.

Intuitively, in the initial phase of the algorithm, the stochastic gradient term dominates, and therefore, SGLD corresponds to a variant
of stochastic gradient ascent. In the later phase, the update rule is
dominated by the injected noise *ϵ*, and will effectively be Langevin
dynamics. SGLD transitions smoothly between the two phases.

Under additional assumptions, SGLD is guaranteed to converge for
decreasing learning rates *η* *t* = Θ ( *t* *[−]* [1] [/] [3] ) . [11] SGLD does not use the ac- 11 Maxim Raginsky, Alexander Rakhlin,
and Matus Telgarsky. Non-convex learning via stochastic gradient langevin dynamics: a nonasymptotic analysis. In
*Conference* *on* *Learning* *Theory*, pages
1674–1703. PMLR, 2017; and Pan Xu,
Jinghui Chen, Difan Zou, and Quanquan
Gu. Global convergence of langevin dynamics based algorithms for nonconvex
optimization. *Advances in Neural Informa-*
*tion Processing Systems*, 31, 2018

136 probabilistic artificial intelligence

ceptance step from Metropolis-Hastings as asymptotically, SGLD corresponds to Langevin dynamics and the Metropolis-Hastings rejection
probability goes to zero for a decreasing learning rate.

*6.4* *Hamiltonian Monte Carlo*

HMC SG-HMC

**sampling**


Figure 6.5: A commutative diagram of
optimization algorithms. Langevin dynamics (LD) is the non-stochastic variant
of SGLD.


GD / Mom.

GD


SGD / Mom.

**stochastic optimization**

SGD


As SGLD and MALA can be seen as a sampling-based analogue of
SGD, a similar analogue for (stochastic) gradient descent with momentum is the (stochastic gradient) *Hamiltonian Monte Carlo* (HMC)
algorithm, which we discuss in the following. [12] 12 Tianqi Chen, Emily Fox, and Carlos
Guestrin. Stochastic gradient hamilto
We have seen that if we want to sample from a distribution nian monte carlo. In *International con-*

–
*ference on machine learning*, pages 1683
1691. PMLR, 2014


We have seen that if we want to sample from a distribution


*p* ( ***x*** ) ∝ exp ( *−* *f* ( ***x*** ))


with energy function *f*, we can construct a Markov chain whose distribution converges to *p* . We have also seen that for this approach to
work, the chain must move through all areas of significant probability
with reasonable speed.

If one is faced with a distribution *p* which is multimodal (i.e., that
has several “peaks”), one has to ensure that the chain will explore all
modes, and can therefore “jump between different areas of the space”.

So in general, *local* updates are doomed to fail. Methods such as
Metropolis-Hastings with Gaussian proposals, or even Langevin Monte
Carlo might face this issue, as they do not jump to distant areas of the

markov chain monte carlo methods 137

state space with significant acceptance probability. It will therefore
take a long time to move from one peak to another.

The HMC algorithm is an instance of Metropolis-Hastings which uses
momentum to propose distant points that conserve energy, with high
acceptance probability. The general idea of HMC is to *lift* samples ***x***
to a higher-order space by considering an auxiliary variable ***y*** with
the same dimension as ***x*** . We also lift the distribution *p* to a distribution on the ( ***x***, ***y*** ) -space by defining a distribution *p* ( ***y*** *|* ***x*** ) and setting
*p* ( ***x***, ***y*** ) = . *p* ( ***y*** *|* ***x*** ) *p* ( ***x*** ) . It is common to pick *p* ( ***y*** *|* ***x*** ) to be a Gaussian
with zero mean and variance *m* ***I*** . Hence,

*p* ( ***x***, ***y*** ) ∝ exp *−* [1] 2 *[−]* *[f]* [ (] ***[x]*** [)] . (6.48)
� 2 *m* *[∥]* ***[y]*** *[∥]* [2] �


Physicists might recognize the above as the canonical distribution of a
Newtonian system if one takes ***x*** as the position and ***y*** as the momentum. *H* ( ***x***, ***y*** ) = . 2 1 *m* *[∥]* ***[y]*** *[∥]* 2 [2] [+] *[ f]* [ (] ***[x]*** [)] [ is called the] *[ Hamiltonian]* [. HMC then]
takes a step in this higher-order space according to the Hamiltonian
dynamics, [13] 13 That is, HMC follows the trajectory of
these dynamics for some time.
*d* ***x*** *d*


*d* ***x*** *d* ***y***

*dt* [=] ***[ ∇]*** ***[y]*** *[ H]* [,] *dt*


*dt* [=] *[ −]* ***[∇]*** ***[x]*** *[ H]* [,] (6.49)


reaching some new point ( ***x*** *[′]*, ***y*** *[′]* ) and *projecting* back to the state space
by selecting ***x*** *[′]* as the new sample. This is illustrated in fig. 6.6.

*y*


Figure 6.6: Illustration of Hamiltonian
Monte Carlo. Shown is the contour plot
of a distribution *p*, which is a mixture of
two Gaussians, in the ( *x*, *y* ) -space.
First, the initial point in the state
space is lifted to the ( *x*, *y* ) -space. Then,
we move according to Hamiltonian dynamics and finally project back onto the
state space.


2

1

0

*−* 1

*−* 2


|Col1|move acc. to Hamiltonian dynamics project lift start|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||new proposal||||||||
||||||||||
||||||||||


*−* 3 *−* 2 *−* 1 0 1 2 3

*x*


In an implementation of this algorithm, one has to solve eq. (6.49) numerically rather than exactly. Typically, this is done using the *Leapfrog*
*method*, which for a step size *τ* computes

***y*** ( *t* + *[τ]* / 2 ) = ***y*** ( *t* ) *−* *[τ]* 2 ***[∇]*** ***[x]*** *[ f]* [ (] ***[x]*** [(] *[t]* [))] (6.50a)

138 probabilistic artificial intelligence

***x*** ( *t* + *τ* ) = ***x*** ( *t* ) + *[τ]* (6.50b)

*m* ***[y]*** [(] *[t]* [ +] *[ τ]* [/] [2] [)]

***y*** ( *t* + *τ* ) = ***y*** ( *t* + *[τ]* / 2 ) *−* *[τ]* 2 ***[∇]*** ***[x]*** *[ f]* [ (] ***[x]*** [(] *[t]* [ +] *[ τ]* [))] [.] (6.50c)

Then, one repeats this procedure *L* times to arrive at a point ( ***x*** *[′]*, ***y*** *[′]* ) . To
correct for the resulting discretization error, the proposal ***x*** *[′]* is either accepted or rejected in a final Metropolis-Hastings acceptance step with
acceptance probability

. *′* *′*
*α* ( ***x*** *[′]* *|* ***x*** ) = min *{* 1, exp ( *H* ( ***x***, ***y*** ) *−* *H* ( ***x***, ***y*** )) *}* . (6.51)


To summarize, Markov chain Monte Carlo methods use a Markov

chain to approximately sample from an intractable distribution. Note
that unlike for variational inference, the convergence of many methods can be guaranteed. Moreover, for log-concave distributions (e.g.,
with Bayesian logistic regression), the underlying Markov chain converges quickly to the stationary distribution. In general, the convergence (mixing time) may be slow, meaning that, in practice, accuracy
and efficiency have to be traded.


### *7* *Bayesian Deep Learning*

*7.1* *Artificial Neural Networks*

In practice, it is often seen that models perform better when labels
may nonlinearly depend on the inputs. One widely used family of
nonlinear functions are *artificial “deep” neural networks*, [1] 1 In the following, we will refrain from
using the characterizations “artificial”

.
***f*** : **R** *[d]* *→* **R** *[k]*, ***f*** ( ***x*** ; ***θ*** ) = ***φ*** ( ***W*** *L* ***φ*** ( ***W*** *L* *−* 1 ( *· · ·* ***φ*** ( ***W*** 1 ***x*** )))) (7.1) and “deep” for better readability.

where ***θ*** = [ . ***W*** 1, . . ., ***W*** *L* ] is a vector of *weights* (written as matrices
***W*** *l* *∈* **R** *[n]* *[l]* *[×]* *[n]* *[l]* *[−]* [1] ) [2] and *φ* : **R** *→* **R** is a component-wise nonlinear func- 2 where *n* 0 = *d* and *n* *L* = *k*
tion. Thus, a deep neural network can be seen as nested (“deep”)
linear functions composed with nonlinearities.


*w* [(] [2] [)]
1,1

*w* *n* [(] [2] 2 [)], *n* 1

###### x 1 x 2 ... *x d*


*w* [(] [1] [)]
1, 1

*w* [(] [1] [)]
*n* 1, *d*

###### ν [(] [1] [)]

1
###### ν [(] [1] [)]

2
###### ... ν n [(] [1] 1 [)]

###### ν [(] [2] [)]

1
###### ν [(] [2] [)]

2
###### ... ν n [(] [2] 2 [)]


Figure 7.1: Computation graph of a neural network with two hidden layers.

###### f 1 ... *f k*


**input layer** **hidden layer 1** **hidden layer 2** **output layer**

A neural network can be visualized by a *computation graph* . An example for such a computation graph is given in fig. 7.1. The columns of
the computation graph are commonly called *layers*, whereby the leftmost column is the *input layer*, the right-most column is the *output*
*layer*, and the remaining columns are the *hidden layers* . The inputs are
(as we have previously) referred to as ***x*** = [ . *x* 1, . . ., *x* *d* ] . The outputs
(i.e., vertices of the output layer) are often referred to as *logits* and

140 probabilistic artificial intelligence

named ***f*** = [ . *f* 1, . . ., *f* *k* ] . The *activations* of an individual (hidden) layer
*l* of the neural network are described by

***ν*** [(] *[l]* [)] = . ***φ*** ( ***W*** *l* ***ν*** ( *l* *−* 1 ) ) (7.2)

where ***ν*** [(] [0] [)] = ***x*** . The activation of the *i* -th node is *ν* *i* [(] *[l]* [)] = ***ν*** [(] *[l]* [)] ( *i* ) .

*7.1.1* *Activation Functions*

The non-linearity *φ* is called an *activation function* . The following two
activation functions are particularly common:

1. The *hyperbolic tangent* (Tanh) is defined as

Tanh ( *z* ) = . exexp p ( ( *zz* ) ) + *−* expex p ( ( *−−* *zz* ) ) *[∈]* [(] *[−]* [1, 1] [)] [.] (7.3)

Tanh is a scaled and shifted variant of the sigmoid function (5.10)
which we have previously seen in the context of logistic regression
as Tanh ( *z* ) = 2 *σ* ( 2 *z* ) *−* 1.

2. The *rectified linear unit* (ReLU) is defined as

ReLU ( *z* ) = . max *{* *z*, 0 *} ∈* [ 0, ∞ ) . (7.4)

In particular, the ReLU activation function leads to “sparser” gradients as it selects the halfspace of inputs with positive sign. Moreover, the gradients of ReLU do not “vanish” as *z* *→±* ∞ which can
lead to faster training.

*7.1.2* *Classification*

Although we mainly focus on regression, neural networks can equally
well be used for classification. If we want to classify inputs into *c* separate classes, we can simply construct a neural network with *c* outputs,
***f*** = [ *f* 1, . . ., *f* *c* ], and normalize them into a probability distribution.
Often, the *softmax function* is used for normalization,

*σ* *i* ( ***f*** ) = . ex p ( *f* *i* ) (7.5)
∑ *[c]* *j* = 1 [exp] [(] *[ f]* *[j]* [)]


Tanh ( *x* )

1

0

*−* 1

*−* 2 0 2

*x*

ReLU ( *x* )

3

2

1

0

*−* 2 0 2

*x*

Figure 7.2: The Tanh and ReLU activation functions, respectively.

bayesian deep learning 141


where *σ* *i* ( ***f*** ) corresponds to the probability mass of class *i* . Note that
the softmax corresponds to a Gibbs distribution with energies *−* ***f*** . In
particular, exercise 6.25 implies that the softmax is the maximum entropy distribution over *c* classes satisfying for a given input ***x*** of class
*i* with logits ***f*** that **E** *j* *∼* *σ* ( ***f*** ) � *f* *j* � = *f* *i* .



*7.1.3* *Loss Functions*

We will study neural networks under the lens of supervised learning
(cf. section 1.2) where we are provided some independently-sampled
(noisy) data *D* = *{* ( ***x*** *i*, ***y*** *i* ) *}* *i* *[n]* = 1 [generated according to an unknown]
process ( ***x***, ***y*** ) *∼* *p*, which we wish to approximate.

Upon initialization, the network does generally not approximate this
process well, so a key element of deep learning is “learning” a parameterization ***θ*** that is a good approximation. To this end, one typically
considers a loss function *ℓ* ( ***θ*** ; ***y*** ) which quantifies the approximation
error of the network outputs ***f*** ( ***x*** ; ***θ*** ) . In the classical setting of regression, i.e., *y* *∈* **R** and *k* = 1, *ℓ* is often taken to be the *(empirical) mean*
*squared error*,


*ℓ* mse ( ***θ*** ; *D* ) = . 1

*n*


*n*
###### ∑ ( f ( x i ; θ ) − y i ) [2] . (7.6)

*i* = 1


Analogously to example 2.6, minimizing mean squared error corresponds to maximum likelihood estimation under a Gaussian likeli
hood.

In the setting of classification where ***y*** *∈{* 0, 1 *}* *[c]* is a one-hot encoding of class membership, [3] it is instead common to interpret the out- 3 That is, exactly one component of ***y*** is 1
puts of a neural network as probabilities akin to our discussion in and all others are 0, indicating to which
class the given example belongs.
section 7.1.2. We denote by *q* ***θ*** ( *· |* ***x*** ) the resulting probability distribution over classes with PMF [ *σ* 1 ( ***f*** ( ***x*** ; ***θ*** )), . . ., *σ* *c* ( ***f*** ( ***x*** ; ***θ*** ))], and aim to
find ***θ*** such that *q* ***θ*** ( ***y*** *|* ***x*** ) *≈* *p* ( ***x***, ***y*** ) . In this context, it is common to

142 probabilistic artificial intelligence

minimize the cross-entropy,


H [ *p* *∥* *q* ***θ*** ] = **E** ( ***x***, ***y*** ) *∼* *p* [ *−* log *q* ***θ*** ( ***y*** *|* ***x*** )] using the definition of cross-entropy

*n* (5.37)

*≈−* [1] ***x***


*n*


*n*
###### ∑ log q θ ( y i | x i )

*i* = 1


(7.7) using Monte Carlo sampling


� �� �
= . *ℓ* ce ( ***θ*** ; *D* )

which can be understood as minimizing the surprise about the training
data under the model. *ℓ* ce is called the *cross-entropy loss* . Disregarding
the constant 1/ *n*, we can rewrite the cross-entropy loss as

*n*
###### ∝ − ∑ log q θ ( y i | x i ) = ℓ nll ( θ ; D ) (7.8)

*i* = 1

Recall that *ℓ* nll ( ***θ*** ; *D* ) is the *negative log-likelihood* of the training data,
and thus, empirically minimizing cross-entropy can equivalently be
interpreted as maximum likelihood estimation. [4] Furthermore, recall 4 We have previously seen this in re
from exercise 5.4 that for a two-class classification problem the cross- mark 5.29. Note that minimizing the

cross-entropy, H [ *p* *∥* *q* ***θ*** ], is equivalent to

entropy loss is equivalent to the logistic loss. minimizing forward-KL, KL ( *p* *∥* *q* ***θ*** ) .

*7.1.4* *Backpropagation*

A crucial property of neural networks is that they are differentiable.
That is, we can compute gradients ***∇*** ***θ*** *ℓ* of ***f*** with respect to the parameterization of the model ***θ*** = ***W*** 1: *L* and some loss function *ℓ* ( ***f*** ; ***y*** ) .
Being able to obtain these gradients efficiently allows for “learning” a
particular function from data using first-order optimization methods.

The algorithm for computing gradients of a neural network is called
*backpropagation* and is essentially a repeated application of the chain
rule. Note that using the chain rule for every path through the network
is computationally infeasible, as this quickly leads to a combinatorial
explosion as the number of hidden layers is increased. The key insight
of backpropagation is that we can use the *feed-forward* structure of our
neural network to memoize computations of the gradient, yielding a
linear time algorithm.

Obtaining gradients by backpropagation is often called *automatic dif-*
*ferentiation (auto-diff)* .

Computing the exact gradient for each data point is still fairly expensive when the size of the neural network is large. Typically, stochastic

bayesian deep learning 143


gradient descent is used to obtain unbiased gradient estimates using
batches of only *m* of the *n* data points, where *m* *≪* *n* .

*7.2* *Bayesian Neural Networks*

How can we adapt our Bayesian approach to neural networks? We use
the same strategy which we already used for Bayesian logistic regression, we impose a Gaussian prior on the weights,

***θ*** *∼N* ( **0**, *σ* *p* [2] ***[I]*** [)] [.] (7.9)

Similarly, we can use a Gaussian likelihood to describe how well the
data is described by the model ***f***,

***y*** *|* ***x***, ***θ*** *∼N* ( ***f*** ( ***x*** ; ***θ*** ), *σ* *n* [2] [)] [.] (7.10)

Thus, instead of considering weights as point estimates which are
learned exactly, *Bayesian neural networks* learn a distribution over the
weights of the network. In principle, other priors and likelihoods can
be used, yet Gaussians are typically chosen due to their closedness
properties, which we have seen in section 1.6 and many times since.

*7.2.1* *Heteroscedastic Noise*

Note that eq. (7.10) uses the scalar parameter *σ* *n* [2] [to model the aleatoric]
uncertainty (label noise), similarly to how we modeled the label noise
in Bayesian linear regression and Gaussian processes. Such a noise

model is called *homoscedastic* as the noise is assumed to be uniform

across the domain. In many settings, however, the noise is inherently
*heteroscedastic* . That is, the noise varies depending on the input and
which “region” of the domain the input is from. This behavior is
visualized in fig. 7.4.

There is a natural way of modeling heteroscedastic noise with Bayesian
neural networks. We use a neural network with two outputs *f* 1 and *f* 2
and define

*y* *|* ***x***, ***θ*** *∼N* ( *µ* ( ***x*** ; ***θ*** ), *σ* [2] ( ***x*** ; ***θ*** )) where (7.11)

*µ* ( ***x*** ; ***θ*** ) = . *f* 1 ( ***x*** ; ***θ*** ), (7.12)

*σ* [2] ( ***x*** ; ***θ*** ) = . exp ( *f* 2 ( ***x*** ; ***θ*** )) . (7.13)

Hereby, we exponentiate *f* 2 to ensure non-negativity of the variance.

*7.3* *Maximum A Posteriori Inference*

Let us first consider MAP estimation in the context of neural networks.
Recall that learning the MAP estimate still corresponds to a point estimate of the weights, so we forgo modeling the epistemic uncertainty.


*x* 1

*x* 2

...

*x* *d*


*ν* 1 [(] [1] [)]
*ν* 2 [(] [1] [)]

...

*ν* *n* [(] [1] 1 [)]


*ν* 1 [(] [2] [)]
*ν* 2 [(] [2] [)]

...

*ν* *n* [(] [2] 2 [)]


*f* 1
...

*f* *k*


Figure 7.3: Bayesian neural networks
model a distribution over the weights of
a neural network.

*y*

*x*

Figure 7.4: Illustration of data with variable (heteroscedastic) noise. The noise
increases as the inputs increase in magnitude. The noise-free function is shown

in black.

144 probabilistic artificial intelligence

We have seen in eq. (1.95) that the MAP estimate of the weights is
obtained as follows,

*n*

ˆ
###### θ MAP = arg max log p ( θ ) + ∑ log p ( y i | x i, θ ) .

***θ*** *i* = 1

Recall from eq. (1.136) that for an isotropic Gaussian prior we have the
regularizer,

log *p* ( ***θ*** ) = *−* *[λ]* 2 [+] [ const] (7.14)

2 *[∥]* ***[θ]*** *[∥]* [2]


where *λ* = . 1 / *σ* *p* 2 is the rate of weight decay. 5 For the Gaussian likeli- 5 This is analogous to our discussion of

hood, we have weight decay in the context of ridge re
gression (cf. eq. (2.17)). We do not have

log *p* ( *y* *i* *|* ***x*** *i*, ***θ*** ) = log *N* ( *y* *i* ; *µ* ( ***x*** *i* ; ***θ*** ), *σ* [2] ( ***x*** *i* ; ***θ*** )) the homoscedastic noiseuse a heteroscedastic noise model. *σ* *n* [2] [here, as we]


hood, we have


log *p* ( *y* *i* *|* ***x*** *i*, ***θ*** ) = log *N* ( *y* *i* ; *µ* ( ***x*** *i* ; ***θ*** ), *σ* [2] ( ***x*** *i* ; ***θ*** ))


1 *−* [(] *[y]* *[i]* *[−]* *[µ]* [(] ***[x]*** *[i]* [;] ***[ θ]*** [))] [2]

2 *πσ* [2] ( ***x*** *i* ; ***θ*** ) 2 *σ* [2] ( ***x*** *i* ; ***θ*** )


1

=
log
�2 *πσ* [2]


note that the normalizing constant
2 *σ* [2] ( ***x*** *i* ; ***θ*** ) depends on the noise model!


.
�

(7.15)


1

=
log
*√* 2 *π*

� �� �
const


*−* [1]

2


log *σ* [2] ( ***x*** *i* ; ***θ*** ) + [(] *[y]* *[i]* *[−]* *[µ]* [(] ***[x]*** *[i]* [;] ***[ θ]*** [))] [2]
� *σ* [2] ( ***x*** *i* ; ***θ*** )


Hence, the model can either explain the label *y* *i* by an accurate model
*µ* ( ***x*** *i* ; ***θ*** ) or by a large variance *σ* [2] ( ***x*** *i* ; ***θ*** ), yet, it is penalized for choosing large variances. Intuitively, this allows to attenuate losses for some
data points by attributing them to large variance (when no model reflecting all data points simultaneously can be found). Note that this
interpretation coincides precisely with the notion of heteroscedastic

noise.

As we have already seen in the context of Bayesian linear regression
(and ridge regression), using a Gaussian prior is equivalent to applying weight decay, [6] 6 Recall that weight decay regularizes
weights by shrinking them towards zero.
***∇*** log *p* ( ***θ*** ) = *−* *[λ]* 2 [=] *[ −]* *[λ]* ***[θ]*** [.] (7.16)

2 ***[∇]*** *[∥]* ***[θ]*** *[∥]* [2]

Using gradient ascent, we obtain the following update rule,


***θ*** *←* ***θ*** ( 1 *−* *λη* *t* ) + *η* *t*


*n*
###### ∑ ∇ log p ( y i | x i, θ ) . (7.17)

*i* = 1


The gradients of the likelihood can be obtained using automatic dif
ferentiation.

*7.4* *Approximate Inference*

Naturally, we want to understand the epistemic uncertainty of our
model too. However, learning and inference in Bayesian neural networks are generally intractable (even when using a Gaussian prior

bayesian deep learning 145

and likelihood) when the noise is not assumed to be homoscedastic
and known. [7] Thus, we are led to the techniques of approximate infer- 7 In this case, the conjugate prior to

ence, which we discussed in the previous two chapters. a Gaussian likelihood is not a Gaus
sian. See, e.g., `https://en.wikipedia.`
`org/wiki/Conjugate` `[_]` `prior` .

*7.4.1* *Variational Inference*


As we have discussed in chapter 5, we can apply black box stochas
tic variational inference which — in the context of neural networks

— is also known as *Bayes by Backprop* . As variational family, we use
the family of independent Gaussians which we have already encountered in example 5.18. [8] Recall the fundamental objective of variational 8 Independent Gaussians are useful be
inference (5.42), cause they can be encoded using only

2 *d* parameters, which is crucial when the
size of the neural network is large.


inference (5.42),


arg min KL ( *q* *∥* *p* ( *· |* ***x*** 1: *n*, *y* 1: *n* ))
*q* *∈Q*


= arg max *L* ( *q*, *p* ; *D* ) using eq. (5.64)
*q* *∈Q*

= arg max **E** ***θ*** *∼* *q* [ log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** )] *−* KL ( *q* *∥* *p* ( *·* )) . using eq. (5.61d)
*q* *∈Q*

*·*
The KL-divergence KL ( *q* *∥* *p* ( )) can be expressed in closed-form for
Gaussians. [9] Recall that we can obtain unbiased gradient estimates of 9 see eq. (5.50)
the expectation using the reparameterization trick (5.80),

**E** ***θ*** *∼* *q* [ log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** )] = **E** ***ϵ*** *∼N* ( **0**, ***I*** ) � log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, ***θ*** ) *|* ***θ*** = ***Cϵ*** + ***µ*** �

where ***C*** is the square root of ***Σ*** . As ***Σ*** is the diagonal matrix diag *i* *∈* [ *d* ] *{* *σ* *i* [2] *[}]* [,]
we have that ***C*** = diag *i* *∈* [ *d* ] *{* *σ* *i* *}* . The gradients of the likelihood can be
obtained using backpropagation.

We can now perform approximate inference using the variational posterior *q* ***λ***,


*p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, ***y*** 1: *n* ) = *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ) *p* ( ***θ*** *|* ***x*** 1: *n*, *y* 1: *n* ) *d* ***θ*** using the sum rule (1.10) and product
� rule (1.14)
= **E** ***θ*** *∼* *p* ( *·|* ***x*** 1: *n*, *y* 1: *n* ) [ *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** )] interpreting the integral as an
expectation over the posterior
*≈* **E** *θ* *∼* *q* ***λ*** [ *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** )] approximating the posterior with the

*m* variational posterior *q* ***λ***

*≈* [1] *[⋆]* ***x*** *[⋆]* ***θ*** [(] *[i]* [)] 18


*m*


*m*
###### ∑ p ( y [⋆] | x [⋆], θ [(] [i] [)] ) (7.18) using Monte Carlo sampling

*i* = 1


for independent samples ***θ*** [(] *[i]* [)] [ iid] *∼* *q* ***λ***,


= [1]

*m*


*m*
###### ∑ N ( y [⋆] ; µ ( x [⋆] ; θ [(] [i] [)] ), σ [2] ( x [⋆] ; θ [(] [i] [)] )) . (7.19) using the neural network

*i* = 1


Intuitively, variational inference in Bayesian neural networks can be
interpreted as averaging the predictions of multiple neural networks
drawn according to the variational posterior *q* ***λ*** .

146 probabilistic artificial intelligence

Using the Monte Carlo samples ***θ*** [(] *[i]* [)], we can also estimate the mean of
our predictions,


**E** [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ] *≈* [1]

*m*


*m*
###### ∑ µ ( x [⋆] ; θ [(] [i] [)] ) = . µ ( x ⋆ ), (7.20)

*i* = 1


and the variance of our predictions,

Var [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ] = **E** ***θ*** �Var *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ] � + Var ***θ*** � **E** *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** ] �


. using the law of total variance (1.50)


Recall from eq. (2.6) that the first term of the law of total variance
corresponds to the aleatoric uncertainty of the data and the second
term corresponds to the epistemic uncertainty of the model. We can
approximate them using the Monte Carlo samples ***θ*** [(] *[i]* [)],


Var [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ] *≈* [1]

*m*


*m*
###### ∑ σ [2] ( x [⋆] ; θ [(] [i] [)] )

*i* = 1


(7.21)


1
+
*m* *−* 1


*m*
###### ∑ ( µ ( x [⋆] ; θ [(] [i] [)] ) − µ ( x [⋆] )) [2]

*i* = 1


using a sample mean (1.69) and sample variance (1.70).

*7.4.2* *Markov Chain Monte Carlo*

As we have discussed in chapter 6, an alternative method to approximating the full posterior distribution is to sample from it directly. By
the ergodic theorem (6.27), we can use any of the discussed sampling
strategies to obtain samples ***θ*** [(] *[t]* [)] such that


*p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, ***y*** 1: *n* ) *≈* [1]

*T*


*T*
###### ∑ p ( y [⋆] | x [⋆], θ [(] [t] [)] ) . see (6.28)

*t* = 1


Here, we omit the offset *t* 0 which is commonly used to avoid the
“burn-in” period for simplicity.

Typically, we cannot afford to store all *T* samples of models. Thus,
we need to summarize the iterates. [10] One approach is to keep track 10 That is, combine the individual samof *m snapshots* of weights [ ***θ*** [(] [1] [)], . . ., ***θ*** [(] *[m]* [)] ] according to some schedule ples of weights ***θ*** [(] *[i]* [)] .
and use those for inference (e.g., by averaging the predictions of the
corresponding neural networks). This approach of sampling a subset
of some data is generally called *subsampling* .


Another approach is to summarize (that is, approximate) the weights
using sufficient statistics (e.g., a Gaussian). [11] In other words, we learn 11 A statistic is *sufficient* for a family of

the Gaussian approximation, probability distributions if the samples

from which it is calculated yield no more
information than the statistic with re
***θ*** *∼N* ( ***µ***, ***Σ*** ), where (7.22a) spect to the learned parameters.


the Gaussian approximation,


***θ*** *∼N* ( ***µ***, ***Σ*** ), where (7.22a)


***µ*** = . 1

*T*


*T*
###### ∑ θ [(] [i] [)], (7.22b) using a sample mean (1.69)

*i* = 1

***Σ*** = . 1
*T* *−* 1


bayesian deep learning 147

*T*
###### ∑ ( θ [(] [i] [)] − µ )( θ [(] [i] [)] − µ ) [⊤], (7.22c) using a sample variance (1.70)

*i* = 1


using sample means and sample (co)variances. This can be implemented efficiently using running averages of the first and second mo
ments,

1 1
***µ*** *←* and ***A*** *←* (7.23)
*T* + 1 [(] *[T]* ***[µ]*** [ +] ***[ θ]*** [)] *T* + 1 [(] *[T]* ***[A]*** [ +] ***[ θθ]*** *[⊤]* [)]

upon observing the fresh sample ***θ*** . ***Σ*** can easily be calculated from
these moments,


*T*
***Σ*** = (7.24) using the characterization of sample
*T* *−* 1 [(] ***[A]*** *[ −]* ***[µµ]*** *[⊤]* [)] [.]
variance in terms of estimators of the
first and second moment (1.71)


(7.24)


To predict, we can sample weights ***θ*** from the learned Gaussian. This
approach is known as the *stochastic weight averaging-Gaussian* (SWAG)

method.



wider optima and better generalization”
(Izmailov et al.).

*7.4.3* *Dropout Regularization*


We will now discuss two approximate inference techniques that are tailored to neural networks. The first is *dropout regularization* (also called
*Monte Carlo dropout* ). Traditionally, dropout regularization randomly
omits vertices of the computation graph during training. The key idea
that we will present here is to interpret dropout regularization as performing variational inference.

Suppose that we omit a vertex of the computation graph with proba

Figure 7.5: Illustration of dropout regularization. Some vertices of the computation graph are randomly omitted.



*x* 1

*x* 2

...

*x* *d*





*f* 1
...

*f* *k*

148 probabilistic artificial intelligence

bility *p* . Then our variational posterior is given as

*d*
###### q ( θ | λ ) = . ∏ q j ( θ j | λ j ) (7.25)

*j* = 1

where *d* is the number of weights of the neural network and

*q* *j* ( *θ* *j* *|* *λ* *j* ) = . *pδ* 0 ( *θ* *j* ) + ( 1 *−* *p* ) *δ* *λ* *j* ( *θ* *j* ) . (7.26)

Here, *δ* *α* is the Dirac delta function with point mass at *α* . [13] The vari- 13 see example 1.20
ational parameters ***λ*** correspond to the “normal” weights of the network. Intuitively, the variational posterior says that the *j* -th weight has
value 0 with probability *p* and value *λ* *j* with probability 1 *−* *p* . This is
visualized in fig. 7.6.

Crucially, for the interpretation of dropout regularization as variational inference to be valid, we also need to perform dropout regularization during inference,

*p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, ***y*** 1: *n* ) *≈* **E** ***θ*** *∼* *q* ***λ*** [ *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** )]


*≈* [1]

*m*


*m*
###### ∑ p ( y [⋆] | x [⋆], θ [(] [i] [)] ) (7.27) using Monte Carlo sampling

*i* = 1


where ***θ*** [(] *[i]* [)] [ iid] *∼* *q* ***λ*** are independent samples. In other words, we average the distribution of *m* neural networks for each of which we randomly “drop out” weights. Observe that this coincides with our earlier discussion of variational inference for Bayesian neural networks in
eq. (7.18).

*7.4.4* *Probabilistic Ensembles*

We have seen that variational inference in the context of Bayesian neural networks can be interpreted as averaging the predictions of *m* neural networks drawn according to the variational posterior.

A natural adaptation of this idea is to immediately learn the weights of
*m* neural networks. The idea is to randomly choose *m* training sets by
sampling uniformly from the data with replacement. Then, using our
analysis from section 7.3, we obtain *m* MAP estimates of the weights
***θ*** [(] *[i]* [)], yielding the approximation

*p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, ***y*** 1: *n* ) = **E** ***θ*** *∼* *p* ( *·|* ***x*** 1: *n*, *y* 1: *n* ) [ *p* ( *y* *[⋆]* *|* ***x*** *[⋆]*, ***θ*** )]


*q* *j*

|qj|Col2|Col3|
|---|---|---|
|1 −p p|||
||||



0 *λ* *j*

*θ*
*j*

Figure 7.6: Interpretation of dropout
regularization as variational inference.
The only coordinates where the variational posterior *q* *j* has positive probability are 0 and *λ* *j* .


*≈* [1]

*m*


*m*
###### ∑ p ( y [⋆] | x [⋆], θ [(] [i] [)] ) . (7.28) using bootstrapping

*i* = 1


Note that this is not equivalent to Monte Carlo sampling, although it
looks very similar. The key difference is that this approach does not

bayesian deep learning 149

sample from the true posterior distribution *p*, but instead from the empirical posterior distribution ˆ *p* given the (re-sampled) MAP estimates.
Intuitively, this can be understood as the difference between sampling
from a distribution *p* directly (Monte Carlo sampling) versus sampling
from an approximate (empirical) distribution ˆ *p* (corresponding to the
training data), which itself is constructed from samples of the true
distribution *p* . This approach is known as *bootstrapping* and plays a
central role in model-free reinforcement learning. We will return to
this concept in section 11.3.1.

*7.5* *Calibration*

A key challenge of Bayesian deep learning (and also other Bayesian
methods) is the calibration of models. We say that a model is *well-*
*calibrated* if its confidence coincides with its accuracy across many predictions. Consider a classification model that predicts that the label
of a given input belongs to some class with probability 80%. If the
model is well-calibrated, then the prediction is correct about 80% of
the time. In other words, during calibration, we adjust the probability

estimation of the model.

We will first mention two methods of estimating the calibration of a
model, namely the marginal likelihood and reliability diagrams. Then,
in section 7.5.3, we survey commonly used heuristics for empirically
improving the calibration.

*7.5.1* *Marginal Likelihood*

A popular method (which we already encountered multiple times) is
to use the marginal likelihood (evidence) of a validation set ***x*** [val] 1: *m* [of]
size *m* given the training set ***x*** [train] 1: *n* of size *n* for estimating the model
calibration. Here, the evidence can be understood as describing how
well the validation set is described by the model trained on the training
set. We obtain,

log *p* ( *y* [val] 1: *m* *[|]* ***[ x]*** [val] 1: *m* [,] ***[ x]*** 1: [train] *n* [,] *[ y]* [train] 1: *n* [)]

= log *p* ( *y* [val] 1: *m* *[|]* ***[ x]*** [val] 1: *m* [,] ***[ θ]*** [)] *[p]* [(] ***[θ]*** *[ |]* ***[ x]*** 1: [train] *n* [,] *[ y]* [train] 1: *n* [)] *[ d]* ***[θ]*** using the sum rule (1.10) and product
� rule (1.14)
*≈* log *p* ( *y* [val] 1: *m* *[|]* ***[ x]*** [val] 1: *m* [,] ***[ θ]*** [)] *[q]* ***[λ]*** [(] ***[θ]*** [)] *[ d]* ***[θ]*** approximating with the variational
�
posterior

*m*
###### = log ∏ p ( y [val] i | x [val] i, θ ) q λ ( θ ) d θ (7.29) using the independence of the data
� *i* = 1

150 probabilistic artificial intelligence

The resulting integrals are typically very small which leads to numerical instabilities. Therefore, it is common to maximize a lower bound

to the marginal likelihood instead,


�

�


= log **E** ***θ*** *∼* *q* ***λ***


*m*
###### ∏ p ( y [val] i | x [val] i, θ )
� *i* = 1


= log **E** ***θ*** *∼* *q* ***λ*** *p* ( *y* *i* *|* ***x*** *i*, ***θ*** ) interpreting the integral as an

*i* = 1 expectation over the variational

*m* posterior
###### ≥ E θ ∼ q λ ∑ log p ( y [val] i | x [val] i, θ ) (7.30) using Jensen’s inequality (5.33)


*m*
###### ∑ log p ( y [val] i | x [val] i, θ )
� *i* = 1


(7.30) using Jensen’s inequality (5.33)


*≈* [1]

*k*


*k*
###### ∑

*j* = 1


*m*
###### ∑ log p ( y [val] i | x [val] i, θ [(] [j] [)] ) (7.31) using Monte Carlo sampling

*i* = 1


for independent samples ***θ*** [(] *[j]* [)] [ iid] *∼* *q* ***λ*** .

*7.5.2* *Reliability Diagrams*


Reliability diagrams take a frequentist perspective to estimate the calibration of a model. For simplicity, we assume a calibration problem
with two classes, 1 and *−* 1 (similarly to logistic regression). [14] 14 Reliability diagrams generalize beyond this restricted example.
We group the predictions of a validation set into *M* interval bins of
size [1] / *M* according to the class probability predicted by the model,
**P** ( *Y* *i* = 1 *|* ***x*** *i* ) . We then compare within each bin, how often the model
thought the inputs belonged to the class (confidence) with how often
the inputs actually belonged to the class (frequency). Formally, we 1.0
define *B* *m* as the set of samples falling into bin *m* and let 0.8


1.0


0.8


freq ( *B* *m* ) = . 1 **1** *{* *Y* *i* = 1 *}* (7.32)
###### | B m | i ∈ [∑] B m

be the proportion of samples in bin *m* that belong to class 1 and let

conf ( *B* *m* ) = . 1 **P** ( *Y* *i* = 1 *|* ***x*** *i* ) (7.33)
###### | B m | i ∈ [∑] B m

be the average confidence of samples belonging to class 1 within the

bin *m* .

Thus, a model is well calibrated if freq ( *B* *m* ) *≈* conf ( *B* *m* ) for each bin
*m* *∈* [ *M* ] . There are two common metrics of calibration that quantify
how “close” a model is to being well calibrated.

1. The *expected calibration error* (ECE) is the average deviation of a
model from perfect calibration,


*M*
###### ℓ ECE = . ∑

*m* = 1


*|* *B* *m* *|*

*|* freq ( *B* *m* ) *−* conf ( *B* *m* ) *|* (7.34)
*n*


0.6

0.4

0.2

0.0

0.00 0.25 0.50 0.75 1.00

conf

1.0

0.8

0.6

0.4

0.2

0.0

0.00 0.25 0.50 0.75 1.00

conf

Figure 7.7: Examples of reliability diagrams with ten bins. A perfectly calibrated model approximates the diagonal
dashed red line. The first reliability diagram shows a well-calibrated model. In
contrast, the second reliability diagram
shows an overconfident model.


where *n* is the size of the validation set.

2. The *maximum calibration error* (MCE) is the maximum deviation of
a model from perfect calibration among all bins,

*ℓ* MCE = . max *m* *∈* [ *M* ] *[|]* [freq] [(] *[B]* *[m]* [)] *[ −]* [conf] [(] *[B]* *[m]* [)] *[|]* [ .] (7.35)

*7.5.3* *Improving Calibration*

We now survey a few heuristics which can be used empirically to improve model calibration.

1. *Histogram binning* assigns a calibrated score *q* *m* = . freq ( *B* *m* ) to each
bin during validation. Then, during inference, we return the calibrated score *q* *m* of the bin corresponding to the prediction of the

model.

2. *Isotonic regression* extends histogram binning by using variable bin

.
boundaries. We find a piecewise constant function ***f*** = [ *f* 1, . . ., *f* *M* ]
that minimizes the bin-wise squared loss,


*n*
###### ∑ 1 { a m ≤ P ( Y i = 1 | x i ) < a m + 1 } ( f m − y i ) [2]

*i* = 1


min
*M*, ***f***, ***a***


*M*
###### ∑

*m* = 1


(7.36a)

subject to 0 = *a* 1 *≤· · · ≤* *a* *M* + 1 = 1, (7.36b)

*f* 1 *≤· · · ≤* *f* *M* (7.36c)

where ***f*** are the calibrated scores and ***a*** = [ . *a* 1, . . ., *a* *M* + 1 ] are the
bin boundaries. We then return the calibrated score *f* *m* of the bin
corresponding to the prediction of the model.
3. *Platt scaling* adjusts the logits *z* *i* of the output layer to

*q* *i* = . *σ* ( *az* *i* + *b* ) (7.37)

and then learns parameters *a*, *b* *∈* **R** to maximize the likelihood.
4. *Temperature scaling* is a special and widely used instance of Platt
scaling where *a* = . 1 / *T* and *b* = . 0 for some temperature scalar

*T* *∈* **R**,


. *z* *i*
*q* *i* = *σ*
� *T*


. (7.38)
�


bayesian deep learning 151

1

0

A B C

1

0

A B C

Figure 7.8: Illustration of temperature
scaling for a classifier with three classes.
On the top, we have a prediction with
a high temperature, yielding a very uncertain prediction (in favor of class *A* ).
Below, we have a prediction with a low
temperature, yielding a prediction that
is strongly in favor of class *A* . Note that
the ranking ( *A* *≻* *C* *≻* *B* ) is preserved.


Intuitively, for a larger temperature *T*, the probability is distributed
more evenly among the classes (without changing the ranking),
yielding a more uncertain prediction. In contrast, for a lower
temperature *T*, the probability is concentrated more towards the
top choices, yielding a less uncertain prediction. As seen in exercise 6.25, temperature scaling can be motivated as tuning the mean

of the softmax distribution.

152 probabilistic artificial intelligence


part II
## *Sequential Decision Making*

### *8* *Active Learning*

By now, we have seen that probabilistic machine learning is very useful for estimating the uncertainty in our models (epistemic uncertainty) and in the data (aleatoric uncertainty). We have been focusing on the setting of supervised learning where we are given a set
*D* = *{* ( ***x*** *i*, *y* *i* ) *}* *i* *[n]* = 1 [of labeled data, yet we often encounter settings]
where we have only little data and acquiring new data is costly.

In this chapter — and in the following chapter on Bayesian optimization — we will discuss how one can use uncertainty to effectively collect more data. In other words, we want to figure out where in the
domain we should sample to obtain the most useful information.



*8.1* *Conditional Entropy*

We begin by introducing the notion of conditional entropy. Recall that
the entropy H [ **X** ] of a random vector **X** can be interpreted as our average surprise when observing realizations ***x*** *∼* **X** . Thus, entropy can
be considered as a quantification of the uncertainty about a random
vector (or equivalently, its distribution). [1] 1 We discussed entropy extensively in
section 5.3.

156 probabilistic artificial intelligence

A natural extension is to consider the entropy of **X** given the occurrence of an event corresponding to another random variable (e.g.,
**Y** = ***y*** for a random vector **Y** ),

.
H [ **X** *|* **Y** = ***y*** ] = **E** ***x*** *∼* *p* ( ***x*** *|* ***y*** ) [ *−* log *p* ( ***x*** *|* ***y*** )] . (8.1)

Instead of averaging over the surprise of samples from the distribution
*p* ( ***x*** ) (like the entropy H [ **X** ] ), this quantity simply averages over the
surprise of samples from the conditional distribution *p* ( ***x*** *|* ***y*** ) .

**Definition 8.2** (Conditional entropy) **.** The *conditional entropy* of a random vector **X** given the random vector **Y** is defined as

H [ **X** *|* **Y** ] = . **E** ***y*** *∼* *p* ( ***y*** ) [ H [ **X** *|* **Y** = ***y*** ]] (8.2)

= **E** ( ***x***, ***y*** ) *∼* *p* ( ***x***, ***y*** ) [ *−* log *p* ( ***x*** *|* ***y*** )] . (8.3)

Intuitively, the conditional entropy of **X** given **Y** describes our average surprise about realizations of **X** given a particular realization of **Y**,
averaged over all such possible realizations of **Y** . In other words, conditional entropy corresponds to the expected remaining uncertainty in
**X** after we observe **Y** . Note that, in general, H [ **X** *|* **Y** ] *̸* = H [ **Y** *|* **X** ] .

It is crucial to stress the difference between H [ **X** *|* **Y** = ***y*** ] and the conditional entropy H [ **X** *|* **Y** ] . The former simply corresponds to a Bayesian
update of our uncertainty in **X** after we have observed the realization
***y*** *∼* **Y** . In contrast, conditional entropy *predicts* how much uncertainty
will remain about **X** (in expectation) after we *will* observe **Y** .

**Definition 8.3** (Joint entropy) **.** One can also define the *joint entropy* of
random vectors **X** and **Y**,

.
H [ **X**, **Y** ] = **E** ( ***x***, ***y*** ) *∼* *p* ( ***x***, ***y*** ) [ *−* log *p* ( ***x***, ***y*** )], (8.4)

as the combined uncertainty about **X** and **Y** . Observe that joint entropy
is symmetric.

This gives the *chain rule for entropy*,

H [ **X**, **Y** ] = H [ **Y** ] + H [ **X** *|* **Y** ] (8.5) using the product rule (1.14) and the
definition of conditional entropy (8.2)

= H [ **X** ] + H [ **Y** *|* **X** ] . (8.6) using symmetry of joint entropy

That is, the joint entropy of **X** and **Y** is given by the uncertainty about
**X** and the additional uncertainty about **Y** given **X** . Moreover, this also
yields *Bayes’ rule for entropy*,

H [ **X** *|* **Y** ] = H [ **Y** *|* **X** ] + H [ **X** ] *−* H [ **Y** ] . (8.7) using the chain rule for entropy (8.5)
twice

A very intuitive property of entropy is its monotonicity: when conditioning on additional observations the entropy can never increase,

H [ **X** *|* **Y** ] *≤* H [ **X** ] . (8.8)

Colloquially, this property is also called the *“information never hurts”*
*principle* . We will derive a proof in the following section.

*8.2* *Mutual Information*

Recall that our fundamental objective is to reduce entropy, as this corresponds to reduced uncertainty in the variables, which we want to
predict. Thus, we are interested in how much information we “gain”
about the random vector **X** by choosing to observe a random vector **Y** .
By our interpretation of conditional entropy from the previous section,
this is described by the following quantity.

**Definition 8.4** (Mutual information, MI) **.** The *mutual information* of **X**
and **Y** (also known as the *information gain* ) is defined as

I ( **X** ; **Y** ) = . H [ **X** ] *−* H [ **X** *|* **Y** ] (8.9)

= H [ **X** ] + H [ **Y** ] *−* H [ **X**, **Y** ] . (8.10)

In words, we subtract the uncertainty left about **X** after observing **Y**
from our initial uncertainty about **X** . This measures the reduction in
our uncertainty in **X** (as measured by entropy) upon observing **Y** . Unlike conditional entropy, it follows from the definition that mutual information is symmetric. That is,

I ( **X** ; **Y** ) = I ( **Y** ; **X** ) . (8.11)


active learning 157

Figure 8.1: Information gain. The first
graph shows the prior. The second
graph shows a selection of samples with
a large information gain (large reduction
in uncertainty). The third graph shows a
selection of samples with a small information gain (small reduction in uncertainty).

H [ **X** ] H [ **X**, **Y** ] H [ **Y** ]







Thus, the mutual information between **X** and **Y** can be understood as

the approximation error (or information loss) when assuming that **X**
and **Y** are independent.


Figure 8.2: Relationship between mutual
information and entropy, expressed as a
Venn diagram.

158 probabilistic artificial intelligence

In particular, using Gibbs’ inequality (cf. exercise 5.20), this relationship shows that I ( **X** ; **Y** ) *≥* 0 with equality when **X** and **Y** are independent, and also proves the *information never hurts principle* (8.8) as

0 *≤* I ( **X** ; **Y** ) = H [ **X** ] *−* H [ **X** *|* **Y** ] .




originates from ***ϵ***



2 [log] � ( 2 *πe* ) *[d]* det� *σ* *n* [2] ***[I]*** �� using the entropy of Gaussians (5.32)







**Definition 8.7** (Conditional mutual information) **.** The *conditional mu-*
*tual information* of **X** and **Y** given **Z** is defined as

I ( **X** ; **Y** *|* **Z** ) = . H [ **X** *|* **Z** ] *−* H [ **X** *|* **Y**, **Z** ] . (8.15)

= H [ **X**, **Z** ] + H [ **Y**, **Z** ] *−* H [ **Z** ] *−* H [ **X**, **Y**, **Z** ] (8.16) using the relationship of joint and
conditional entropy (8.5)

= I ( **X** ; **Y**, **Z** ) *−* I ( **X** ; **Z** ) . (8.17)

Thus, the conditional mutual information corresponds to the reduction
of uncertainty in **X** when observing **Y**, given we already observed **Z** .
It also follows that conditional mutual information is symmetric, that

is,

I ( **X** ; **Y** *|* **Z** ) = I ( **Y** ; **X** *|* **Z** ) . (8.18)


active learning 159




*Mutual information as an optimization objective* Following our introduction of mutual information, it is natural to answer the question “where
should I collect data?” by saying “wherever mutual information is
maximized”. More concretely, assume we are given a set *X* of possible observations of *f*, where *y* ***x*** denotes a single such observation at

160 probabilistic artificial intelligence

***x*** *∈X*,

*y* ***x*** = . *f* ***x*** + *ϵ* ***x***, (8.21)

*f* ***x*** = . *f* ( ***x*** ), and *ϵ* ***x*** is some zero-mean noise. For a set of observations
*S* = *{* ***x*** 1, . . ., ***x*** *n* *}*, we can write ***y*** *S* = ***f*** *S* + ***ϵ*** where









, and ***ϵ*** *∼N* ( **0**, *σ* *n* [2] ***[I]*** [)] [.]


, ***f*** *S* = .


*f* ***x*** 1

...

*f* ***x*** *n*


***y*** *S* = .







*y* ***x*** 1

...

*y* ***x*** *n*


Note that both ***y*** *S* and ***f*** *S* are random vectors. Our goal is then to find
a subset *S* *⊆X* of size *n* maximizing the information gain between
our model *f* and ***y*** *S* .

This yields the maximization objective,

*I* ( *S* ) = . I ( ***f*** *S* ; ***y*** *S* ) = H [ ***f*** *S* ] *−* H [ ***f*** *S* *|* ***y*** *S* ] . (8.22)

Here, H [ ***f*** *S* ] corresponds to the uncertainty about ***f*** *S* before obtaining the observations ***y*** *S* and H [ ***f*** *S* *|* ***y*** *S* ] corresponds to the uncertainty
about ***f*** *S* after obtaining the observations ***y*** *S* .

Observe that picking a subset of points *S* *⊆X* from the domain *X* is
a combinatorial problem. That is to say, we are optimizing a function
over discrete sets. In general, such combinatorial optimization problems tend to be very difficult. It can be shown that maximizing mutual
information is *N P* -hard.

*8.3* *Submodularity of Mutual Information*

We will look at optimizing mutual information in the following section. First, we want to introduce the notion of submodularity which is
important in the analysis of discrete functions.

**Definition 8.9** (Marginal gain) **.** Given a (discrete) function *F* : *P* ( *X* ) *→*
**R**, the *marginal gain* of ***x*** *∈X* given *A* *⊆X* is defined as

∆ *F* ( ***x*** *|* *A* ) = . *F* ( *A* *∪{* ***x*** *}* ) *−* *F* ( *A* ) . (8.23)

Intuitively, the marginal gain describes how much “adding” the addi
tional ***x*** to *A* increases the value of *F* .


active learning 161




That is, when maximizing mutual information, the marginal gain corresponds to the difference between the uncertainty after observing *y* *A*
and the entropy of the noise H [ *ϵ* ***x*** ] . Altogether, the marginal gain represents the reduction in uncertainty by observing *{* ***x*** *}* .

**Definition 8.11** (Submodularity) **.** A (discrete) function *F* : *P* ( *X* ) *→* **R**
is *submodular* iff for any ***x*** *∈X* and any *A* *⊆* *B* *⊆X* it satisfies

*F* ( *A* *∪{* ***x*** *}* ) *−* *F* ( *A* ) *≥* *F* ( *B* *∪{* ***x*** *}* ) *−* *F* ( *B* ) . (8.26)

Equivalently, using our definition of marginal gain, we have that *F* is
submodular iff for any ***x*** *∈X* and any *A* *⊆* *B* *⊆X*,

∆ *F* ( ***x*** *|* *A* ) *≥* ∆ *F* ( ***x*** *|* *B* ) . (8.27)

That is, “adding” ***x*** to the smaller set *A* yields more marginal gain
than adding ***x*** to the larger set *B* . In other words, the function *F* has
“diminishing returns”. In this way, submodularity can be interpreted
as a notion of “concavity” for discrete functions.

**Definition 8.12** (Monotone submodularity) **.** A function *F* : *P* ( *X* ) *→* **R**
is called *monotone* iff for any *A* *⊆* *B* *⊆X* it satisfies

*F* ( *A* ) *≤* *F* ( *B* ) . (8.28)

If *F* is also submodular, then *F* is called *monotone submodular* .

**Theorem 8.13.** *The objective I is monotone submodular.*

*Proof.* We fix arbitrary subsets *A* *⊆* *B* *⊆X* and any ***x*** *∈X* . We have,






Figure 8.3: Monotone submodularity.
The effect of “adding” ***x*** to the smaller
set *A* is larger than the effect of adding
***x*** to the larger set *B* .


*I* is submodular *⇐⇒* ∆ *I* ( ***x*** *|* *A* ) *≥* ∆ *I* ( ***x*** *|* *B* ) by submodularity in terms of marginal
gain (8.27)
*⇐⇒* H [ *y* ***x*** *|* ***y*** *A* ] *−* H [ *ϵ* ***x*** ] *≥* H [ *y* ***x*** *|* ***y*** *B* ] *−* H [ *ϵ* ***x*** ] using eq. (8.25)

*⇐⇒* H [ *y* ***x*** *|* ***y*** *A* ] *≥* H [ *y* ***x*** *|* ***y*** *B* ] . H [ *ϵ* ***x*** ] cancels

Due to the “information never hurts” principle (8.8) of entropy and as
*A* *⊆* *B*, this is always true. Moreover,

*I* is monotone *⇐⇒* *I* ( *A* ) *≤* *I* ( *B* ) by the definition of monotinicity (8.28)

*⇐⇒* I ( ***f*** *A* ; ***y*** *A* ) *≤* I ( ***f*** *B* ; ***y*** *B* ) using the definition of *I* (8.22)

*⇐⇒* I ( ***f*** *B* ; ***y*** *A* ) *≤* I ( ***f*** *B* ; ***y*** *B* ) using I ( ***f*** *B* ; ***y*** *A* ) = I ( ***f*** *A* ; ***y*** *A* ) as
***y*** *A* *⊥* ***f*** *B* *|* ***f*** *A*
*⇐⇒* H [ ***f*** *B* ] *−* H [ ***f*** *B* *|* ***y*** *A* ] *≤* H [ ***f*** *B* ] *−* H [ ***f*** *B* *|* ***y*** *B* ] using the definition of MI (8.9)

162 probabilistic artificial intelligence

*⇐⇒* H [ ***f*** *B* *|* ***y*** *A* ] *≥* H [ ***f*** *B* *|* ***y*** *B* ], H [ ***f*** *B* ] cancels

which is also satisfied due to the “information never hurts” principle
(8.8).

*8.4* *Maximizing Mutual Information*

*8.4.1* *Uncertainty Sampling*

As we cannot efficiently pick a set *S* *⊆X* to maximize mutual information, a natural approach is to maximize mutual information greedily.
That is, we pick the locations ***x*** 1 through ***x*** *n* individually by greedily
finding the location with the maximal mutual information.

At time *t*, when we have already picked *S* *t* = *{* ***x*** 1, . . ., ***x*** *t* *}*, we need to
solve the following optimization problem,

***x*** *t* + 1 = . arg max ∆ *I* ( ***x*** *|* *S* *t* ) (8.29)
***x*** *∈X*

= arg max I ( *f* ***x*** ; *y* ***x*** *|* ***y*** *S* *t* ) . (8.30) using eq. (8.24)
***x*** *∈X*

Note that *f* ***x*** and *y* ***x*** are univariate random variables. Thus, using our

formula for the mutual information of conditional linear Gaussians

(8.14), we can simplify to,


= arg max
***x*** *∈X*


1 1 + *[σ]* *t* [2] [(] ***[x]*** [)]
2 [log] � *σ* *n* [2]


(8.31)
�


where *σ* *t* [2] [(] ***[x]*** [)] [ is the (remaining) variance at] ***[ x]*** [ after observing] *[ S]* *[t]* [. As-]
suming the label noise is independent of ***x*** (i.e., homoscedastic),

= arg max *σ* *t* [2] [(] ***[x]*** [)] [.] (8.32)
***x*** *∈X*

Therefore, if *f* is modeled by a Gaussian and we assume homoscedastic noise, greedily maximizing mutual information corresponds to simply picking the point ***x*** with the largest variance. This strategy is also
called *uncertainty sampling* .






active learning 163



such that ∆ *F* ( ***x*** *|* *S* *t* ) is maximized (8.29)









*y*

*8.4.2* *Heteroscedastic Noise*

Uncertainty sampling is clearly problematic if the noise is heteroscedastic. If there are a particular set of inputs with a large aleatoric uncertainty dominating the epistemic uncertainty, uncertainty sampling
will continuously choose those points even though the epistemic un*x* *[⋆]*
certainty will not be reduced substantially. See fig. 8.4, for an example. *x*


Looking at eq. (8.31) suggests a natural fix. Instead of only considering the epistemic uncertainty *σ* *t* [2] [(] ***[x]*** [)] [, we can also consider the aleatoric]


Figure 8.4: Uncertainty sampling with
heteroscedastic noise. The epistemic uncertainty of the model is shown in a dark
gray. The aleatoric uncertainty of the
data is shown in a light gray. Uncertainty sampling would repeatedly pick
points around *x* *[⋆]* as they maximize the
*total* uncertainty.

164 probabilistic artificial intelligence

uncertainty *σ* *n* [2] [(] ***[x]*** [)] [ by modeling heteroscedastic noise, yielding]


= arg max
� ***x*** *∈X*


*σ* *t* [2] [(] ***[x]*** [)]
(8.34)
*σ* *n* [2] ( ***x*** ) [.]


***x*** *t* + 1 = . arg max
***x*** *∈X*


1 1 + *[σ]* *t* [2] [(] ***[x]*** [)]
2 [log] � *σ* *n* [2] ( ***x*** )


Thus, we choose locations that trade large epistemic uncertainty with
large aleatoric uncertainty. Ideally, we find a location where the epistemic uncertainty is large, and the aleatoric uncertainty is low, which
promises a significant reduction of uncertainty around this location.

*8.4.3* *Classification*

While we focused on regression, one can apply active learning also for
other settings, such as (probabilistic) classification. In this setting, for
any input ***x***, a model produces a categorical distribution over labels
*y* ***x*** . [3] Here, uncertainty sampling corresponds to selecting samples that 3 see section 1.2
maximize the entropy of the predicted label *y* ***x***,

***x*** *t* + 1 = . arg max H [ *y* ***x*** *|* ***x*** 1: *t*, *y* 1: *t* ] . (8.35)
***x*** *∈X*

The entropy of a categorical distribution is simply a finite sum of
weighted surprise terms.

Figure 8.5: Uncertainty sampling in classification. The area with high uncertainty (as measured by entropy) is highlighted in yellow. The shown figures display each a sequence of model updates,
each after one new observation. In the
left figure, the classes are well-separated
and uncertainty is dominated by epistemic uncertainty, whereas in the right
figure the uncertainty is dominated by
noise. In the latter case, if we mostly
choose points ***x*** in the area of highest
uncertainty (i.e., close to the decision
boundary) to make observations, the label noise results in frequently changing
This approach generally leads to sampling points that are close to the models.
decision boundary. Often, the uncertainty is mainly dominated by label noise rather than epistemic uncertainty, and hence, we do not learn
much from our observations. This is a similar problem to the one we
encountered with uncertainty sampling in the setting of heteroscedas
tic noise.

This naturally suggests distinguishing between the aleatoric and epistemic uncertainty of the model *f* (parameterized by ***θ*** ). To this end,
mutual information can be used similarly as we have done with uncertainty sampling for regression,

***x*** *t* + 1 = . arg max I ( ***θ*** ; *y* ***x*** *|* ***x*** 1: *t*, *y* 1: *t* ) (8.36)
***x*** *∈X*

active learning 165

= arg max I ( *y* ***x*** ; ***θ*** *|* ***x*** 1: *t*, *y* 1: *t* ) using symmetry (8.11)
***x*** *∈X*

= arg max H [ *y* ***x*** *|* ***x*** 1: *t*, *y* 1: *t* ] *−* H [ *y* ***x*** *|* ***θ***, ***x*** 1: *t*, *y* 1: *t* ] using the definition of mutual
***x*** *∈X* information (8.9)
= arg max H [ *y* ***x*** *|* ***x*** 1: *t*, *y* 1: *t* ] *−* **E** ***θ*** *|* ***x*** 1: *t*, *y* 1: *t* [ H [ *y* ***x*** *|* ***θ***, ***x*** 1: *t*, *y* 1: *t* ]] (8.37) using the definition of conditional
***x*** *∈X* entropy (8.2)


= arg max H [ *y* ***x*** *|* ***x*** 1: *t*, *y* 1: *t* ]
***x*** *∈X* � �� �
entropy of
predictive posterior


*−* **E** ***θ*** *|* ***x*** 1: *t*, *y* 1: *t* H� [ *y* �� ***x*** *|* ***θ*** � ]
entropy of
likelihood


. (8.38) using the definition of entropy (5.29)
and assuming *y* ***x*** *⊥* ***x*** 1: *t*, *y* 1: *t* *|* ***θ***


The first term measures the entropy of the averaged prediction while
the second term measures the average entropy of predictions. Thus,
the first term looks for points where the average prediction is not confident. In contrast, the second term penalizes points where many of
the sampled models are not confident about their prediction, and thus
looks for points where the models are confident in their predictions.
This identifies those points ***x*** where the models *disagree* about the label
*y* ***x*** (that is, each model is “confident” but the models predict different labels). For this reason, this approach is known as *Bayesian active*
*learning by disagreement* (BALD).

Note that the second term of the difference acts as a regularizer when
compared to eq. (8.35). The second term mirrors our description of
aleatoric uncertainty from section 2.2. Recall that we interpreted aleatoric uncertainty as the average uncertainty for all models. Crucially,
here we use entropy to “measure” uncertainty, whereas previously we
have been using variance. Therefore, intuitively, eq. (8.37) subtracts the
aleatoric uncertainty from the total uncertainty about the label.

Observe that both terms require approximate forms of the posterior
distribution. In chapters 5 and 6, we have seen various approaches

from variational inference and MCMC methods which can be used

here. The first term can be approximated by the predictive distribution of an approximated posterior which is obtained, for example, using variational inference. The nested entropy of the second term is
typically easy to compute, as it corresponds to the entropy of the (discrete) likelihood distribution *p* ( *y* *|* ***x***, ***θ*** ) of the model ***θ*** . The outer
expectation of the second term may be approximated using (approximate) samples from the posterior distribution obtained via variational
inference, MCMC, or some other method.


### *9* *Bayesian Optimization*

Often, obtaining data is costly. In the previous chapter, this led us
to investigate how we can optimally improve our understanding (i.e.,
reduce uncertainty) of the process we are trying to model. However,
purely improving our understanding is often not good enough. In
many cases, we want to use our improving understanding *simultane-*
*ously* to reach certain goals. This is a very common problem in artificial
intelligence and will concern us for the rest of this course. One common instance of this problem is the setting of optimization.

Given some function *f* *[⋆]* : *X →* **R**, suppose we want to find the

arg max *f* *[⋆]* ( ***x*** ) . (9.1)
***x*** *∈X*

Now, contrary to classical optimization, we are interested in the setting
where the function *f* *[⋆]* is unknown to us (like a “black-box”). We are
only able to obtain noisy observations of *f* *[⋆]*,

*y* *t* = *f* *[⋆]* ( ***x*** *t* ) + *ϵ* *t* . (9.2)


***x*** *t*


*y* *t* = *f* *[⋆]* ( ***x*** *t* ) + ***ϵ*** *t*


*f* *[⋆]*

Figure 9.1: Illustration of Bayesian optimization. We pass an input ***x*** *t* into the
unknown function *f* *[⋆]* to obtain noisy observations *y* *t* .


Moreover, these noise-perturbed evaluations are costly to obtain. We
will assume that similar alternatives yield similar results. [1] This allows 1 That is, *f* *⋆* is “smooth”. We will be

us to learn a model of *f* *[⋆]* from relatively few samples. [2] more precise in the subsequent parts of

this chapter. If this were not the case,
optimizing the function without evaluating it everywhere would not be possible.

*9.1* *Exploration-Exploitation Dilemma* Fortunately, many interesting functions

obey by this relatively weak assumption.


us to learn a model of *f* *[⋆]* from relatively few samples. [2]


*9.1* *Exploration-Exploitation Dilemma*


In Bayesian optimization, we want to learn a model of *f* *[⋆]* and use
this model to optimize *f* *[⋆]* simultaneously. These goals are somewhat
contrary. Learning a model of *f* *[⋆]* requires us to explore the input space
while using the model to optimize *f* *[⋆]* requires us to focus on the most
promising well-explored areas. This trade-off is commonly known as
the *exploration-exploitation dilemma*, whereby


2 There are countless examples of this
problem in the “real world”. Instances

are

- drug trials

- chemical engineering — the development of physical products

- recommender systems

- automatic machine learning — automatic tuning of model & hyperpa
rameters

- and many more...

168 probabilistic artificial intelligence

- *exploration* refers to choosing points that are “informative” with respect to the unknown function. For example, points that are far
away from previously observed points (i.e., have high posterior variance); [3] and 3 We explored this topic (with strategies

- *exploitation* refers to choosing promising points where we expect the like uncertainty sampling) in the previous chapter.
function to have high values. For example, points that have a high
posterior mean and a low posterior variance.

In other words, the exploration-exploitation dilemma refers to the challenge of learning enough to understand *f* *[⋆]*, but not learning too much
to lose track of the objective — optimizing *f* *[⋆]* .

It is common to use a so-called *acquisition function* to greedily pick the
next point to sample based on the current model.

*9.2* *Online Learning and Bandits*

Bayesian optimization is closely related to a form of online learning.
In *online learning* we are given a set of possible inputs *X* and an unknown function *f* *[⋆]* : *X →* **R** . We are now asked to choose a sequence
of inputs ***x*** 1, . . . ***x*** *T* online, [4] and our goal is to maximize our cumu- 4 *Online* is best translated as “sequenlative reward ∑ *t* *[T]* = 1 *[f]* *[ ⋆]* [(] ***[x]*** *[t]* [)] [. Depending on what we observe about] *[ f]* *[ ⋆]* [,] tial”. That is, we need to pick ***x*** *t* + 1 based
only on our prior observations *y* 1, . . ., *y* *t* .
there are different variants of online learning. Bayesian optimization
is closest to the so-called (stochastic) “bandit” setting.

*9.2.1* *Multi-Armed Bandits*

The “ *multi-armed bandits* ” (MAB) problem is a classical, canonical formalization of the exploration-exploitation dilemma. In the MAB problem, we are provided with *k* possible actions (arms) and want to maxi
mize our reward online within the time horizon *T* . We do not know the

reward distributions of the actions in advance, however, so we need to

trade learning the reward distribution with following the most promising action. Bayesian optimization can be interpreted as a variant of the
MAB problem where there can be a potentially infinite number of actions (arms), but their rewards are correlated (because of the smoothness of the Gaussian process prior).


There exists a large body of work on this and similar problems in online decision-making. Much of this work develops theory on how to
explore and exploit in the face of uncertainty. The shared prevalence
of the exploration-exploitation dilemma signals a deep connection between online learning and Bayesian optimization (and — as we will
later come to see — reinforcement learning). Many of the approaches
which we will encounter in the context of these topics are strongly
related to methods in online learning.


Figure 9.2: Illustration of a multi-armed
bandit with four arms, each with a different reward distribution. The agent
tries to identify the arm with the most
beneficial reward distribution shown in

green.

bayesian optimization 169


One of the key principles of the theory on multi-armed bandits and
reinforcement learning is the *“optimism in the face of uncertainty” princi-*
*ple*, which suggests that it is a good guideline to explore where we can
hope for the best outcomes.



*9.2.2* *Regret*

The key performance metric in online learning is the regret.

**Definition 9.1** (Regret) **.** The *(cumulative) regret* for a time horizon *T*
associated with choices *{* ***x*** *t* *}* *t* *[T]* = 1 [is defined as]


*T*
###### R T = . ∑

*t* = 1


max *f* *[⋆]* ( ***x*** ) *−* *f* *[⋆]* ( ***x*** *t* )
� ***x*** �

� �� �
*instantaneous regret*


(9.3)


*T*
###### = T max x f [⋆] ( x ) − ∑ f [⋆] ( x t ) . (9.4)

*t* = 1

The regret can be interpreted as the additive loss with respect to the
*static* optimum max ***x*** *f* *[⋆]* ( ***x*** ) .

The goal is to find algorithms that achieve *sublinear* regret,

lim *R* *T* (9.5)
*T* *→* ∞ *T* [=] [ 0.]

Importantly, if we use an algorithm which explores *forever*, e.g., by going to a random point ˜ ***x*** with a constant probability *ϵ* in each round,
then the regret will grow linearly with time. This is because the instantaneous regret is at least *ϵ* ( max ***x*** *f* *[⋆]* ( ***x*** ) *−* *f* *[⋆]* ( ˜ ***x*** )) and non-decreasing.
Conversely, if we use an algorithm which *never* explores, then we
might never find the static optimum, and hence, also incur constant
instantaneous regret in each round, implying that regret grows linearly with time. Thus, achieving sublinear regret requires *balancing*
exploration with exploitation.

170 probabilistic artificial intelligence





Typically, online learning (and Bayesian optimization) consider stationary environments, hence the comparison to the static optimum.
Dynamic environments are studied in *online algorithms* (see metrical
task systems [5], convex function chasing [6], and generalizations of multi
armed bandits to changing reward distributions) and reinforcement
learning. When operating in dynamic environments, other metrics
such as the competitive ratio, [7] which compares against the best *dy-*

*namic* choice, are useful. As we will later come to see in section 13.1 in
the context of reinforcement learning, operating in dynamic environments is deeply connected to a rich field of research called *control* .

*9.3* *Acquisition Functions*

Throughout our description of acquisition functions, we will focus on a
setting where we model *f* *[⋆]* using a Gaussian process which we denote
by *f* . The methods generalize to other means of learning *f* *[⋆]* such
as Bayesian neural networks. The various acquisition functions *F* are
used in the same way as is illustrated in alg. 9.3.

**Al** **g** **orithm 9.3:** Ba y esian o p timization (with GPs)

**1** initialize *f* *∼GP* ( *µ* 0, *k* 0 )

**2** `for` *t* = 1 `to` *T* `do`

**3** choose ***x*** *t* = arg max ***x*** *∈X* *F* ( ***x*** ; *µ* *t* *−* 1, *k* *t* *−* 1 )

**4** sample *y* *t* = *f* ( ***x*** *t* ) + *ϵ* *t*

**5** perform a Bayesian update to obtain *µ* *t* and *k* *t*

One possible acquisition function is uncertainty sampling (8.31), which


5 Metrical task systems are a classical example in online algorithms. Suppose we
are moving in a (finite) decision space
*X* . In each round, we are given a “task”
*f* *t* : *X →* **R** which is more or less costly
depending on our state ***x*** *t* *∈X* . In many
contexts, it is natural to assume that it
is also costly to move around in the decision space. This cost is modeled by a
metric *d* ( *·*, *·* ) on *X* . In *metrical task sys-*
*tems*, we want to minimize our total cost,

*T*
∑ *f* *t* ( ***x*** *t* ) + *d* ( ***x*** *t*, ***x*** *t* *−* 1 ) .
*t* = 1

That is, we want to trade completing our
tasks optimally with moving around in
the state space. Crucially, we do not
know the sequence of tasks *f* *t* in advance. Due to the cost associated with
moving in the decision space, previous
choices affect the future!
6 *Convex function chasing* (or *convex body*
*chasing* ) generalize metrical task systems
to continuous domains *X* . To make
any guarantees about the performance
in these settings, one typically has to
assume that the tasks *f* *t* are convex.
Note that this mirrors our assumption in
Bayesian optimization that similar alternatives yield similar results.

7 To assess the performance in dynamic
environments, we typically compare to
a dynamic optimum. As these problems
are difficult (we are usually not able to
guarantee convergence to the dynamic
optimum), one considers a multiplicative performance metric similar to the
approximation ratio, the *competitive ratio*,

cost ( ALG ) *≤* *α* *·* cost ( OPT ),

where OPT corresponds to the dynamic
optimal choice (in hindsight).

bayesian optimization 171


we discussed in the previous chapter. However, this acquisition function does not at all take into account the objective of maximizing *f* *[⋆]*

and focuses solely on exploration.


*y*


Suppose that our model *f* of *f* *[⋆]* is well-calibrated, in the sense that
the true function lies within its confidence bounds. Consider the best 4
lower bound, that is, the maximum of the lower confidence bound.
Now, if the true function is really contained in the confidence bounds, 2
it must hold that the optimum is somewhere above this best lower
bound. In particular, we can exclude all regions of the domain where 0
the upper confidence bound (the optimistic estimate of the function
value) is lower than the best lower bound. This is visualized in fig. 9.3. 0 2 4


4


2


0


Therefore, we only really care how the function looks like in the regions where the upper confidence bound is larger than the best lower
bound. The key idea behind the methods that we will explore is to
focus exploration on these plausible maximizers.

Note that it is crucial that our uncertainty about *f* reflects the “fit” of

our model to the unknown function. If the model is not well calibrated

or does not describe the underlying function at all, these methods will
perform very poorly. This is where we can use the Bayesian philosophy by imposing a prior belief (that may be conservative).

*9.3.1* *Upper Confidence Bound*

The principle of optimism in the face of uncertainty suggests picking
the point where we can hope for the optimal outcome. In this setting, this corresponds to simply maximizing the *upper confidence bound*
(UCB),

UCB *t* ( ***x*** ) = . *µ* *t* *−* 1 ( ***x*** ) + *β* *t* *σ* *t* *−* 1 ( ***x*** ), (9.8)


Figure 9.3: Optimism in Bayesian optimization. The **unknown function** is

shown in black, our **model** in blue with
gray confidence bounds. The dotted
black line denotes the maximum lower

bound. We can therefore focus our exploration to the yellow regions where
the upper confidence bound is higher
than the maximum lower bound.


4

2

0

4

2


where *σ* *t* ( ***x*** ) = . � *k* *t* ( ***x***, ***x*** ) is the standard deviation at ***x*** and *β* *t* regu
lates how confident we are about our model *f* (i.e., how to choose the
confidence interval).

This acquisition function naturally trades exploitation by preferring a
large posterior mean with exploration by preferring a large posterior
variance. Observe that if we omit the exploitation bias *µ* *t* *−* 1 ( ***x*** ), we
recover uncertainty sampling. [8]

This optimization problem is non-convex in general. However, we can
use approximate global optimization techniques like Lipschitz optimization (in low dimensions) and gradient ascent with random initialization (in high dimensions). Another widely used option is to sample
some random points from the domain, score them according to this
criterion, and simply take the best one.


2.5

2.0

1.5

1.0

0.5

3.5

3.0


2.5

0

2.0

Figure 9.4: Plot of the UCB acquisition
function for *β* = 0.5 and *β* = 2, respectively.

8 Due to the monotonicity of ( *·* ) 2, it does
not matter whether we optimize the variance or standard deviation at ***x*** .

172 probabilistic artificial intelligence

**Theorem 9.4** (Bayesian regret for GP-UCB) **.** *Under the assumption that*
*f* *[⋆]* *∼GP* ( *µ* 0, *k* 0 ) *(i.e., f* *[⋆]* *is a sample from the GP) and if β* *t* *is chosen “cor-*
*rectly”, greedily choosing the upper confidence bound yields average regret*


*R* *T* = *O* [˜]
��


*Tγ* *T* (9.9)
�


*with high probability, where*


*γ* *T* = . max
*S* *⊆X*
*|* *S* *|* = *T*


I ( ***f*** *S* ; ***y*** *S* ) (9.10)


*is the maximum information gain after T rounds.*

*Proof.* Refer to theorem 2 of “Gaussian process optimization in the
bandit setting: No regret and experimental design” (Srinivas et al.).









Observe that if the information gain is sublinear in *T* then we achieve
sublinear regret and, in particular, converge to the true optimum.

**Theorem 9.6** (Information gain of common kernels) **.** *Due to submod-*
*ularity, we have the following bounds on the information gain of common*

*kernels:*

- linear kernel

- Gaussian kernel


*γ* *T* = *O* ( *d* log *T* ), (9.13)


bayesian optimization 173

40

20

0

20 40

*T*

Figure 9.5: Information gain of **inde-**
**pendent**, **linear**, **Gaussian**, and **Matérn**
( *ν* *≈* 0.5) kernels with *d* = 2 (up to constant factors). The kernels with sublinear information gain have strong diminishing returns (due to their strong dependence between “close” points). In
contrast, the independent kernel has no
dependence between points in the domain, and therefore no diminishing returns. Intuitively, the “smoother” the
class of functions modeled by the kernel,
the stronger are the diminishing returns.


*γ* *T* = *O* ( log *T* ) *[d]* [+] [1] [�], (9.14)
�

- Matérn kernel *for ν* *>* [1] 2

*d* 2 *ν*
*γ* *T* = *O* *T* 2 *ν* + *d* ( log *T* ) 2 *ν* + *d* . (9.15)
� �

*Proof.* Refer to theorem 5 of “Gaussian process optimization in the
bandit setting: No regret and experimental design” (Srinivas et al.)
and remark 2 of “On information gain and regret bounds in gaussian
process bandits” (Vakili et al.).






For the Bayesian regret, we assumed that the unknown *f* *[⋆]* was drawn
according to our model which rarely happens in practice. It turns
out that for some class of functions one can still analyze how well the
Bayesian prior matches with reality.

174 probabilistic artificial intelligence

**Theorem 9.8** (Frequentist regret for GP-UCB) **.** *Assuming f* *[⋆]* *∈H* *k* *, we*
*have that with probability at least* 1 *−* *δ*


*y*

3

2

1

0

*−* 1

*x*

Figure 9.6: Re-scaling the confidence
bounds. The dotted gray lines represent
updated confidence bounds.


*R* *T* = *O*
��

*where β* [2] *t* [=] [ 2] *[ ∥]* *[f]* *[ ⋆]* *[∥]* [2] *k* [+] [ 300] *[γ]* *[t]* [ log] [3] [(] *[t]* [/] *[δ]* [)] *[.]*


*Tβ* [2] *T* *[γ]* *[T]*


(9.18)
�


*Proof.* Refer to theorem 3 of “Gaussian process optimization in the
bandit setting: No regret and experimental design” (Srinivas et al.).

Intuitively, to work even if the unknown function *f* *[⋆]* is not contained
in the confidence bounds, we use *β* *t* to re-scale the confidence bounds
to enclose *f* *[⋆]* as shown in fig. 9.6. Here, *β* *t* depends on the information gain of the kernel as well as on the “complexity” of *f* *[⋆]* which is
measured in terms of the norm of the underlying reproducing kernel
Hilbert space *H* *k* .

*9.3.2* *Probability of Improvement*


The *probability of improvement* (PI) picks the point that maximizes the
probability to improve on the running optimum *f* [ˆ], [9] 9 Φ denotes the CDF of the standard normal distribution.


�


PI *t* ( ***x*** ) = . **P** *f* *t* *−* 1 ( ***x*** ) *>* *f* [ˆ] = Φ
� �


*µ* *t* *−* 1 ( ***x*** ) *−* *f* [ˆ]

*σ* *t* *−* 1 ( ***x*** )

�


(9.19) using linear transformations of
Gaussians (1.120)


where we use that *f* *t* ( ***x*** ) *∼N* ( *µ* *t* ( ***x*** ), *σ* *t* [2] [(] ***[x]*** [))] [. Probability of improve-]
ment tends to be biased in favor of exploitation, as it prefers points
with large posterior mean and small posterior variance (which is typically true “close” to the previously observed maximum *f* [ˆ] ).

*9.3.3* *Expected Improvement*

Given a model *f*, the improvement of a point ***x*** over the previously
observed maximum *f* [ˆ] is measured by ( *f* ( ***x*** ) *−* *f* [ˆ] ) + where we use ( *·* ) +
to denote max *{* 0, *·}* . A natural approach is to choose the point which
maximizes the *expected improvement* (EI),


0.4

0.3

0.2


4

2


0.1

0

0.0

0.15

4

0.10

2

0.05

0

0.00

Figure 9.7: Plot of the PI and EI acquisition functions, respectively.

10 Vu Nguyen, Sunil Gupta, Santu Rana,
Cheng Li, and Svetha Venkatesh. Regret
for expected improvement over the bestobserved value and stopping condition.
In *Asian Conference on Machine Learning*,
pages 279–294. PMLR, 2017


EI *t* ( ***x*** ) = . **E** *f* *|* ***x*** 1: *t* *−* 1, *y* 1: *t* *−* 1


( *f* ( ***x*** ) *−* *f* [ˆ] ) + . (9.20)
� �


Intuitively, expected improvement seeks a large expected improvement
(exploitation) while also preferring states with a large variance (exploration). Expected improvement yields the same frequentist regret
bounds (9.18) as UCB. [10]

bayesian optimization 175












*9.3.4* *Thompson Sampling*

We can also interpret the principle of optimism in the face of uncertainty in a slightly different way. Suppose we select the next point
according to the probability that it is optimal (assuming that the posterior distribution is an accurate representation of the uncertainty),


*π* ( ***x*** *|* ***x*** 1: *t*, *y* 1: *t* ) = . **P** *f* *|* ***x*** 1: *t*, *y* 1: *t* *f* ( ***x*** ) = max *f* ( ***x*** *[′]* ) (9.23)

� ***x*** *[′]* �


***x*** *t* + 1 *∼* *π* ( *· |* ***x*** 1: *t*, *y* 1: *t* ) . (9.24)

This approach is called *probability matching* . Probability matching is
exploratory as it prefers points with larger variance (as they automatically have a larger chance of being optimal), but at the same time
exploitative as it effectively discards points with low posterior mean
and low posterior variance. Unfortunately, it is generally difficult to
compute *π* analytically given a posterior.

Instead, it is common to use a sampling-based approximation of *π* .

176 probabilistic artificial intelligence

Observe that the density *π* can be expressed as an expectation,

*π* ( ***x*** *|* ***x*** 1: *t*, *y* 1: *t* ) = **E** *f* *|* ***x*** 1: *t*, *y* 1: *t* � **1** *{* *f* ( ***x*** ) = max *f* ( ***x*** *[′]* ) *}* �, (9.25)
***x*** *[′]*

which we can approximate using Monte Carlo sampling (typically using a single sample),

˜
*≈* **1** *{* *f* [˜] *t* + 1 ( ***x*** ) = max *f* *t* + 1 ( ***x*** *[′]* ) *}* (9.26)
***x*** *[′]*

where *f* [˜] *t* + 1 *∼* *p* ( *· |* ***x*** 1: *t*, *y* 1: *t* ) is a sample from our posterior distribution. Observe that this approximation of *π* coincides with a point
density at the maximizer of *f* [˜] *t* + 1 .

The resulting algorithm is known as *Thompson sampling* . At time *t* + 1,
we sample a function *f* [˜] *t* + 1 *∼* *p* ( *· |* ***x*** 1: *t*, *y* 1: *t* ) from our posterior distribution. Then, we simply maximize *f* [˜] *t* + 1,

***x*** *t* + 1 = . arg max *f* ˜ *t* + 1 ( ***x*** ) . (9.27)
***x*** *∈X*

In many cases, the randomness in the realizations of *f* [˜] *t* + 1 is already
sufficient to effectively trade exploration and exploitation. Similar regret bounds to those of UCB can also be established for Thompson
sampling.

1.0

0.5

0.0

*−* 0.5

*−* 1.0


0.2 0.4 0.6 0.8 1.0

*σ* *t*


0.2 0.4 0.6 0.8 1.0

*σ* *t*


0.2 0.4 0.6 0.8 1.0

*σ* *t*


Figure 9.8: Contour lines of acquisition
functions for varying ∆ *t* = *µ* *t* ( ***x*** ) *−* *f* *[⋆]*

*9.4* *Model Selection* and *σ* *t* . The first graph shows contour
lines of UCB with *β* *t* = 0.75, the second
of PI, and the third of EI.

Selecting a model of *f* *[⋆]* is much harder than in the i.i.d. data setting
of supervised learning. There are mainly the two following dangers,

- the data sets collected in active learning and Bayesian optimization
are *small* ; and

- the data points are selected *dependently* on prior observations.

This leads to a specific danger of overfitting. In particular, due to
feedback loops between the model and the acquisition function, one
may end up sampling the same point repeatedly.

bayesian optimization 177


One approach to reduce the chance of overfitting is the use of hyperpriors which we mentioned previously in section 4.4.2. Another
approach that often works fairly well is to occasionally (according to
some schedule) select points uniformly at random instead of using the
acquisition function. This tends to prevent getting stuck in suboptimal
parts of the state space.


### *10* *Markov Decision Processes*

We will now turn to the topic of probabilistic planning. Planning deals
with the problem of deciding which action an agent should play in
a (stochastic) environment. [1] A key formalism for probabilistic plan- 1 An environment is *stochastic* as opning in *known* environments are so-called Markov decision processes. posed to deterministic, when the outcome of actions is random.
Starting from the next chapter, we will look at reinforcement learning,
which extends probabilistic planning to unknown environments.


Consider the setting where we have a sequence of states ( *X* *t* ) *t* *∈* **N** 0 similarly to Markov chains. But now, the next state *X* *t* + 1 of an agent does
not only depend on the previous state *X* *t* but also depends on the last
action *A* *t* of this agent.

**Definition 10.1** ((Finite) Markov decision process, MDP) **.** A *(finite)*
*Markov decision process* is specified by

- a (finite) set of *states X* = . *{* 1, . . ., *n* *}*,

- a (finite) set of *actions A* = . *{* 1, . . ., *m* *}*,

- *transition probabilities*

*p* ( *x* *[′]* *|* *x*, *a* ) = . **P** � *X* *t* + 1 = *x* *[′]* *|* *X* *t* = *x*, *A* *t* = *a* � (10.1)

which is also called the *dynamics model*, and

- a *reward function r* : *X* *×* *A* *→* **R** which maps the current state *x* and

an action *a* to some reward.

The reward function may also depend on the next state *x* *[′]*, however,
we stick to the above model for simplicity. Also, the reward function
can be random with mean *r* . Observe that *r* induces the sequence of
rewards ( *R* *t* ) *t* *∈* **N** 0, where

*R* *t* = . *r* ( *X* *t*, *A* *t* ), (10.2)

which is sometimes used in the literature instead of *r* .

Crucially, we assume the dynamics model *p* and the reward function
*r* to be known. That is, we operate in a known environment. For now,


Figure 10.1: Directed graphical model of
a Markov decision process with hidden
states *X* *t* and actions *A* *t* .

180 probabilistic artificial intelligence

we also assume that the environment is *fully observable* . In other words,
we assume that our agent knows its current state. In section 10.4, we
discuss how this method can be extended to the partially observable
setting.

Our fundamental objective is to learn how the agent should behave to
optimize its reward. In other words, given its current state, the agent
should decide (optimally) on the action to play. Such a decision map
— whether optimal or not — is called a policy.

**Definition 10.2** (Policy) **.** A *policy* is a function that maps each state
*x* *∈* *X* to a probability distribution over the actions. That is, for any

*t* *>* 0,

*π* ( *a* *|* *x* ) = . **P** ( *A* *t* = *a* *|* *X* *t* = *x* ) . (10.3)

In other words, a policy assigns to each action *a* *∈* *A*, a probability of
being played given the current state *x* *∈* *X* .

We assume that policies are stationary, that is, do not change over time.

Observe that a policy induces a Markov chain ( *X* *t* *[π]* [)] *[t]* *[∈]* **[N]** 0 [with transition]
probabilities,
###### p [π] ( x [′] | x ) = . P � X t [π] + 1 [=] [ x] [′] [ |] [ X] t [π] [=] [ x] � = ∑ π ( a | x ) p ( x [′] | x, a ) . (10.4)

*a* *∈* *A*

This is crucial: if our agent follows a fixed policy (i.e., decision-making
protocol) then the evolution of the process is described fully by a

Markov chain.

As mentioned, we want to maximize the reward. There are many
models of calculating a score from the infinite sequence of rewards
( *R* *t* ) *t* *∈* **N** 0 . For the purpose of our discussion of Markov decision processes and reinforcement learning, we will focus on a very common
reward called discounted payoff.

markov decision processes 181


**Definition 10.4** (Discounted payoff) **.** The *discounted payoff* (also called
*discounted total reward* ) from time *t* is defined as the random variable,

∞
###### G t = . ∑ γ [m] R t + m (10.5)

*m* = 0

where *γ* *∈* [ 0, 1 ) is the *discount factor* .







We now want to understand the effect of the starting state and initial
action on our optimization objective *G* *t* . To analyze this, it is common
to use the following two functions:

**Definition 10.6** (State value function) **.** The *state value function*, [2] 2 Recall that following a fixed policy *π*
induces a Markov chain ( *X* *t* *[π]* [)] *[t]* *[∈]* **[N]** 0 [. We]

*v* *[π]* *t* [(] *[x]* [)] = . **E** *π* [ *G* *t* *|* *X* *t* = *x* ], (10.7) define **E** *π* [ *·* ] = . **E** ( *X* *tπ* [)] *[t]* *[∈]* **[N]** 0 [[] *[·]* []] (10.6)


measures the average discounted payoff from time *t* starting from state

*x* *∈* *X* .

**Definition 10.7** (State-action value function) **.** The *state-action value func-*
*tion* (also called *Q-function* ),

*q* *t* *[π]* [(] *[x]* [,] *[ a]* [)] = . **E** *π* [ *G* *t* *|* *X* *t* = *x*, *A* *t* = *a* ] (10.8)


as an expectation over all possible sequences of states ( *x* *t* ) *t* *∈* **N** 0 within this
Markov chain.

###### = r ( x, a ) + γ ∑ p ( x [′] | x, a ) · v [π] t + 1 [(] [x] [′] [)] [,] (10.9) by expanding the defition of the

*x* *[′]* *∈* *X* discounted payoff (10.5); corresponds to
one step in the induced Markov chain

measures the average discounted payoff from time *t* starting from state
*x* *∈* *X* and with playing action *a* *∈* *A* . In other words, it combines the

immediate return with the value of the next states.

Note that both *v* *[π]* *t* [(] *[x]* [)] [ and] *[ q]* *[π]* *t* [(] *[x]* [,] *[ a]* [)] [ are deterministic scalar-valued func-]

tions.

Because we assumed stationary dynamics, rewards, and policies, the
discounted payoff starting from a given state *x* will be independent of
the start time *t* . Thus, we write *v* *[π]* ( *x* ) = . *v* 0 *π* [(] *[x]* [)] [ and] *[ q]* *[π]* [(] *[x]* [,] *[ a]* [)] = . *q* 0 *π* [(] *[x]* [,] *[ a]* [)]
without loss of generality.

182 probabilistic artificial intelligence

*10.1* *Bellman Expectation Equation*

Let us now see how we can compute the value function,


*v* *[π]* ( *x* ) = **E** *π* [ *G* 0 *|* *X* 0 = *x* ] using the definition of the value
function (10.7)


*X* 0 = *x*
�����


�


= **E** *π*


∞
###### ∑ γ [m] R m
� *m* = 0


*X* 0 = *x*
�����


�


using the definition of the discounted
payoff (10.5)

using linearity of expectation (1.24)


= **E** *π* *γ* [0] *R* 0 *X* 0 = *x* + *γ* **E** *π*
� ��� �


∞
###### ∑ γ [m] R m + 1
� *m* = 0


*X* 1 = *x* *′* *X* 0 = *x*
����� ������


�


= *r* ( *x*, *π* ( *x* )) + *γ* **E** *x* *′*


�


**E** *π*


∞
###### ∑ γ [m] R m + 1
� *m* = 0


= *r* ( *x*, *π* ( *x* )) + *γ* **E** *x* *′* **E** *π* *γ* *R* *m* + 1 *X* 1 = *x* *X* 0 = *x* by simplifying the first expectation and

*m* = 0 conditioning the second expectation on

∞ *X* 1

*′*
###### = r ( x, π ( x )) + γ ∑ p ( x [′] | x, π ( x )) E π ∑ γ [m] R m + 1 X 1 = x expanding the expectation on X 1 and

*x* *[′]* *∈* *X* � *m* = 0 ����� � using conditional independence of the


∞
###### ∑ γ [m] R m + 1
� *m* = 0


*X* 1 = *x* *′*
����� �


= *r* ( *x*, *π* ( *x* )) + *γ* *p* ( *x* *|* *x*, *π* ( *x* )) **E** *π* *γ* *R* *m* + 1 *X* 1 = *x* expanding the expectation on *X* 1 and
*x* *[′]* *∈* *X* *m* = 0 using conditional independence of the

∞ discounted payoff of *X* 0 given *X* 1

*′*
###### = r ( x, π ( x )) + γ ∑ p ( x [′] | x, π ( x )) E π ∑ γ [m] R m X 0 = x shifting the start time of the discounted

*x* *[′]* *∈* *X* � *m* = 0 ����� � payoff using stationarity


∞
###### ∑ γ [m] R m
� *m* = 0


*X* 0 = *x* *′*
����� �


shifting the start time of the discounted
payoff using stationarity

###### = r ( x, π ( x )) + γ ∑ p ( x [′] | x, π ( x )) E π � G 0 | X 0 = x [′] [�] using the definition of the discounted

*x* *[′]* *∈* *X* payoff (10.5)
###### = r ( x, π ( x )) + γ ∑ p ( x [′] | x, π ( x )) · v [π] ( x [′] ) . (10.10) using the definition of the value

*x* *[′]* *∈* *X* function (10.7)

= *r* ( *x*, *π* ( *x* )) + *γ* **E** *x* *′* *|* *x*, *π* ( *x* ) � *v* *[π]* ( *x* *[′]* ) �. (10.11) interpreting the sum as an expectation

This equation is known as the *Bellman expectation equation*, and it shows
a recursive dependence of the value function on itself. The intuition
is clear: the value of the current state corresponds to the reward from
the next action plus the discounted sum of all future rewards obtained
from the subsequent states.

For stochastic policies, the above calculation can be extended to yield,

###### r ( x, a ) + γ ∑ p ( x [′] | x, a ) v [π] ( x [′] )

*x* *[′]* *∈* *X* �

###### v [π] ( x ) = ∑ π ( a | x )

*a* *∈* *A*


�


(10.12)


= **E** *a* *∼* *π* ( *x* ) � *r* ( *x*, *a* ) + *γ* **E** *x* *′* *|* *x*, *a* � *v* *[π]* ( *x* *[′]* ) � [�] (10.13)

= **E** *a* *∼* *π* ( *x* ) [ *q* *[π]* ( *x*, *a* )] . (10.14)

For stochastic policies, by also conditioning on the first action, one can
obtain an analogous equation for the state-action value function,
###### q [π] ( x, a ) = r ( x, a ) + γ ∑ p ( x [′] | x, a ) ∑ π ( a [′] | x [′] ) q [π] ( x [′], a [′] ) (10.15)

*x* *[′]* *∈* *X* *a* *[′]* *∈* *A*

= *r* ( *x*, *a* ) + *γ* **E** *x* *′* *|* *x*, *a* **E** *a* *′* *∼* *π* ( *x* *′* ) � *q* *[π]* ( *x* *[′]*, *a* *[′]* ) �. (10.16)

Note that it does not make sense to consider a similar recursive for
mula for the state-action value function in the setting of deterministic

policies as the action played when in state *x* *∈* *X* is uniquely determined as *π* ( *x* ) . In particular,

*v* *[π]* ( *x* ) = *q* *[π]* ( *x*, *π* ( *x* )) . (10.17)


markov decision processes 183

Figure 10.2: Suppose you are building
a company. The shown MDP models “how to become rich and famous”.

Here, the action *S* is short for *saving* and
the action *A* is short for *advertising* .
Suppose you begin by being “poor
and unknown”. Then, the greedy action (i.e., the action maximizing instantaneous reward) is to save. However,
within this simplified environment, saving when you are poor and unknown
means that you will remain poor and unknown forever. As the potential rewards
in other states are substantially larger,
this simple example illustrates that following the greedy choice is generally not
optimal.




poor,

famous


unknown famous


1 (-1)







1 / 2 (0)


1 / 2 (10)








*10.2* *Policy Evaluation*

Bellman’s expectation equation tells us how we can find the value
function *v* *[π]* of a fixed policy *π* using a system of linear equations!

184 probabilistic artificial intelligence

Using,










, ***r*** *[π]* = .




*r* ( 1, *π* ( 1 ))

...

*r* ( *n*, *π* ( *n* ))




, and



***v*** *[π]* = .

***P*** *[π]* = .












*v* *[π]* ( 1 )

...

*v* *[π]* ( *n* )


*p* ( 1 *|* 1, *π* ( 1 )) *· · ·* *p* ( *n* *|* 1, *π* ( 1 ))

... ... ...

*p* ( 1 *|* *n*, *π* ( *n* )) *· · ·* *p* ( *n* *|* *n*, *π* ( *n* ))





(10.18)







and a little bit of linear algebra, the Bellman expectation equation
(10.10) is equivalent to

***v*** *[π]* = ***r*** *[π]* + *γ* ***P*** *[π]* ***v*** *[π]* (10.19)

*⇐⇒* ( ***I*** *−* *γ* ***P*** *[π]* ) ***v*** *[π]* = ***r*** *[π]*

*⇐⇒* ***v*** *[π]* = ( ***I*** *−* *γ* ***P*** *[π]* ) *[−]* [1] ***r*** *[π]* . (10.20)

Solving this linear system of equations (i.e., performing matrix inversion) takes cubic time in the size of the state space.

*10.2.1* *Fixed-point Iteration*

To obtain an (approximate) solution of ***v*** *[π]*, we can use that it is the
unique fixed-point of the affine mapping ***B*** *[π]* : **R** *[n]* *→* **R** *[n]*,

***B*** *[π]* ***v*** = . ***r*** *π* + *γ* ***P*** *π* ***v*** . (10.21)

Using this fact (which we will prove in just a moment), we can use
fixed-point iteration of ***B*** *[π]* .

**Al** **g** **orithm 10.9:** Fixed- p oint iteration

**1** initialize ***v*** *[π]* (e.g., as **0** )

**2** `for` *t* = 1 `to` *T* `do`

**3** ***v*** *[π]* *←* ***B*** *[π]* ***v*** *[π]* = ***r*** *[π]* + *γ* ***P*** *[π]* ***v*** *[π]*

Fixed-point iteration has computational advantages, for example, for
sparse transitions.

**Theorem 10.10.** ***v*** *[π]* *is the unique fixed-point of* ***B*** *[π]* *.*

*Proof.* It is immediate from Bellman’s expectation equation (10.19) and
the definition of ***B*** *[π]* (10.21) that ***v*** *[π]* is a fixed-point of ***B*** *[π]* . To prove
uniqueness, we will show that ***B*** *[π]* is a contraction.

Let ***v*** *∈* **R** *[n]* and ***v*** *[′]* *∈* **R** *[n]* be arbitrary initial guesses. We use the *ℓ* ∞
space, [3]


markov decision processes 185

3 The *ℓ* ∞ *norm* (also called *supremum*
*norm* ) is defined as



[3] *∥* ***x*** *∥* ∞ = . max *i* *|* ***x*** ( *i* ) *|* . (10.23)

*π* *π* *′* *π* *π* *π* *π* *′*
�� ***B*** ***v*** *−* ***B*** ***v*** �� ∞ [=] �� ***r*** + *γ* ***P*** ***v*** *−* ***r*** *−* *γ* ***P*** ***v*** �� ∞ using the definition of ***B*** *[π]* (10.21)


= *γ* �� ***P*** *π* ( ***v*** *−* ***v*** *′* ) �� ∞

*′* *′* *′*
###### ≤ γ max x ∈ X x [∑] [′] ∈ X p ( x [′] | x, π ( x )) · �� v ( x ) − v ( x ) �� . using the definition of the(10.23), expanding the multiplication, ℓ ∞ norm

and using *|* ∑ *i* *a* *i* *| ≤* ∑ *i* *|* *a* *i* *|*
*′*
*≤* *γ* �� ***v*** *−* ***v*** �� ∞ [.] (10.24) using ∑ *x* *′* *∈* *X* *p* ( *x* *[′]* *|* *x*, *π* ( *x* )) = 1 and
*|* ***v*** ( *x* *[′]* ) *−* ***v*** *[′]* ( *x* *[′]* ) *| ≤∥* ***v*** *−* ***v*** *[′]* *∥* ∞

Thus, by eq. (10.22), ***B*** *[π]* is a contraction and by Banach’s fixed-point
theorem ***v*** *[π]* is its unique fixed-point.

Let ***v*** *t* *[π]* [be the value function estimate after] *[ t]* [ iterations. Then, we have]
for the convergence of fixed-point iteration,

*∥* ***v*** *t* *[π]* *[−]* ***[v]*** *[π]* *[∥]* ∞ [=] �� ***B*** *π* ***v*** *tπ* *−* 1 *[−]* ***[B]*** *[π]* ***[v]*** *[π]* [��] ∞ using the update rule of fixed-point
iteration and ***B*** *[π]* ***v*** *[π]* = ***v*** *[π]*

*π*
*≤* *γ* �� ***v*** *t* *−* 1 *[−]* ***[v]*** *[π]* [��] ∞ using (10.24)

= *γ* *[t]* *∥* ***v*** 0 *[π]* *[−]* ***[v]*** *[π]* *[∥]* ∞ [.] (10.25) by induction

This shows that fixed-point iteration converges to ***v*** *[π]* exponentially
quickly.

*10.3* *Policy Optimization*

Recall that our goal was to find an optimal policy,

*π* *[⋆]* = . arg max **E** *π* [ *G* 0 ] . (10.26)

*π*

We can alternatively characterize an optimal policy as follows: We
define a partial ordering over policies by

*π* *≥* *π* *[′]* *⇐⇒* *·* *v* *[π]* ( *x* ) *≥* *v* *[π]* *[′]* ( *x* ) ( *∀* *x* *∈* *X* ) . (10.27)

186 probabilistic artificial intelligence

*π* *[⋆]* is then simply a policy which is maximal according to this partial
ordering.

It follows that all optimal policies have identical value functions. Subsequently, we use *v* *[⋆]* = . *v* *π* *⋆* and *q* *⋆* = . *q* *π* *⋆* to denote the state value
function and state-action value function arising from an optimal policy, respectively. As an optimal policy maximizes the value of each
state, we have that

*v* *[⋆]* ( *x* ) = max *v* *[π]* ( *x* ), *q* *[⋆]* ( *x*, *a* ) = max *q* *[π]* ( *x*, *a* ) . (10.28)
*π* *π*

Simply optimizing over each policy is not a good idea as there are *m* *[n]*

deterministic policies in total. It turns out that we can do much better.

*10.3.1* *Greedy Policies*

Consider a policy that acts greedily according to the immediate return.
It is fairly obvious that this policy will not perform well because the
agent might never get to high-reward states. But what if someone
could tell us not just the immediate return, but the long-term value of
the states our agent can reach in a single step? If we knew the value of
each state our agent can reach, then we can simply pick the action that
maximizes the expected value. We will make this approach precise in

the next section.

This thought experiment suggests the definition of a greedy policy
with respect to a value function.

**Definition 10.12** (Greedy policy) **.** The *greedy policy* with respect to a
state-action value function *q* is defined as

*π* *q* ( *x* ) = . arg max *q* ( *x*, *a* ) . (10.29)
*a* *∈* *A*

Analogously, we define the *greedy policy* with respect to a state value
function *v*,
###### π v ( x ) = . arg max r ( x, a ) + γ ∑ p ( x [′] | x, a ) · v ( x [′] ) . (10.30)

*a* *∈* *A* *x* *[′]* *∈* *X*


*10.3.2* *Bellman Optimality Equation*

Observe that following the greedy policy *π* *v*, will lead us to a new
value function *v* *[π]* *[v]* . With respect to this value function, we can again
obtain a greedy policy, of which we can then obtain a new value function. In this way, the correspondence between greedy policies and
value functions induces a cyclic dependency, which is visualized in
fig. 10.3.

It turns out that the optimal policy *π* *[⋆]* is a fixed-point of this dependency. This is made precise by the following theorem.

**Theorem 10.14** (Bellman’s theorem) **.** *A policy π* *[⋆]* *is optimal iff it is greedy*
*with respect to its own value function. In other words, π* *[⋆]* *is optimal iff π* *[⋆]* ( *x* )
*is a distribution over the set* arg max *a* *∈* *A* *q* *[⋆]* ( *x*, *a* ) *.*

In particular, if for every state there is a unique action that maximizes the state-action value function, the policy *π* *[⋆]* is deterministic
and unique,

*π* *[⋆]* ( *x* ) = arg max *q* *[⋆]* ( *x*, *a* ) . (10.32)
*a* *∈* *A*

*Proof.* It is a direct consequence of eq. (10.28) that a policy is optimal
iff it is greedy with respect to *q* *[⋆]* .

This theorem confirms our intuition from the previous section that
greedily following an optimal value function is itself optimal. In particular, Bellman’s theorem shows that there always exists an optimal
policy which is deterministic and stationary.

We have seen, that *π* *[⋆]* is a fixed-point of greedily picking the best
action according to its state-action value function. The converse is also

true.

**Corollary 10.15.** *The optimal value functions v* *[⋆]* *and q* *[⋆]* *are a fixed-point of*
*the so-called* Bellman update *,*

*v* *[⋆]* ( *x* ) = max *a* *∈* *A* *[q]* *[⋆]* [(] *[x]* [,] *[ a]* [)] [,] (10.33)


markov decision processes 187

*v* *[π]* induces *π* *v* *π*

*π* *v* induces *v* *[π]* *[v]*

Figure 10.3: Cyclic dependency between
**value function** and **greedy policy** .


= max � *v* *[⋆]* ( *x* *[′]* ) � (10.34) *using the definition of the q-function* (10.9)
*a* *∈* *A* *[r]* [(] *[x]* [,] *[ a]* [) +] *[ γ]* **[E]** *[x]* *[′]* *[|]* *[x]* [,] *[a]*


*q* *[⋆]* ( *x*, *a* ) = *r* ( *x*, *a* ) + *γ* **E** *x* *′* *|* *x*, *a*

*Proof.* It follows from eq. (10.14) that


�max *a* *[′]* *∈* *A* *[q]* *[⋆]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] �. (10.35)


*v* *[⋆]* ( *x* ) = **E** *a* *∼* *π* *⋆* ( *x* ) [ *q* *[⋆]* ( *x*, *a* )] . (10.36)

Thus, as *π* *[⋆]* is greedy with respect to *q* *[⋆]*, *v* *[⋆]* ( *x* ) = max *a* *∈* *A* *q* *[⋆]* ( *x*, *a* ) .

Equation (10.35) follows analogously from eq. (10.15).

188 probabilistic artificial intelligence

These equations are also called the *Bellman optimality equations* . Intuitively, the Bellman optimality equations express that the value of a
state under an optimal policy must equal the expected return for the

best action from that state. Bellman’s theorem is also known as *Bell-*

*man’s optimality principle*, which is a more general concept.


**Bellman’s optimality principle** Bellman’s optimality equations for MDPs
are one of the main settings of Bellman’s optimality principle. However,
Bellman’s optimality principle has many
other important applications, for example in dynamic programming. Broadly
speaking, Bellman’s optimality principle
says that optimal solutions to decision
problems can be decomposed into optimal solutions to sub-problems.








The two perspectives of Bellman’s theorem naturally suggest two separate ways of finding the optimal policy. Policy iteration uses the perspective from eq. (10.32) of *π* *[⋆]* as a fixed-point of the dependency between greedy policy and value function. In contrast, value iteration
uses the perspective from eq. (10.33) of *v* *[⋆]* as the fixed-point of the
Bellman update. Another approach which we will not discuss here is
to use a linear program where the Bellman update is interpreted as a
set of linear inequalities.

*10.3.3* *Policy Iteration*

Starting from an arbitrary initial policy, policy iteration as shown in
alg. 10.17 uses the Bellman expectation equation to compute the value
function of that policy (as we have discussed in section 10.2) and then
chooses the greedy policy with respect to that value function as its

next iterate.

Let *π* *t* be the policy after *t* iterations. We will now show that policy

markov decision processes 189

**Al** **g** **orithm 10.17:** Polic y iteration

**1** initialize *π* (arbitrarily)

**2** `repeat`

**3** compute *v* *[π]*

**4** compute *π* *v* *π*

**5** *π* *←* *π* *v* *π*

**6** `until` `converged`

iteration converges to the optimal policy. The proof is split into two
parts. First, we show that policy iteration improves policies monotonically. Then, we will use this fact to show that policy iteration

converges.

**Lemma 10.18** (Monotonic improvement of policy iteration) **.** *We have,*

 - *v* *[π]* *[t]* [+] [1] ( *x* ) *≥* *v* *[π]* *[t]* ( *x* ) *for all x* *∈* *X; and*

 - *v* *[π]* *[t]* [+] [1] ( *x* ) *>* *v* *[π]* *[t]* ( *x* ) *for at least one x* *∈* *X, unless v* *[π]* *[t]* *≡* *v* *[⋆]* *.*

*Proof.* We consider the Bellman update from (10.33) as the mapping
***B*** *[⋆]* : **R** *[n]* *→* **R** *[n]*,

( ***B*** *[⋆]* ***v*** )( *x* ) = . max *a* *∈* *A* *[q]* [(] *[x]* [,] *[ a]* [)] [,] (10.37)

where *q* is the state-action value function corresponding to the state
value function ***v*** *∈* **R** *[n]* . Recall that after obtaining *v* *[π]* *[t]*, policy iteration
first computes the greedy policy w.r.t. *v* *[π]* *[t]*, *π* *t* + 1 = . *π* *v* *πt*, and then
computes its value function *v* *[π]* *[t]* [+] [1] .

To establish the (weak) monotonic improvement of policy iteration, we
consider a fixed-point iteration (cf. alg. 10.9) of ***v*** *[π]* *[t]* [+] [1] initialized by
***v*** *[π]* *[t]* . We denote the iterates by ˜ ***v*** *t*, in particular, we have that ˜ ***v*** 0 = ***v*** *[π]* *[t]*

and lim *t* *→* ∞ ***v*** ˜ *t* = ***v*** *[π]* *[t]* [+] [1] . [4] First, observe that for the first iteration of 4 using the convergence of fixed-point itfixed-point iteration, eration (10.25)

*v* ˜ 1 ( *x* ) = ( ***B*** *[⋆]* ***v*** *[π]* *[t]* )( *x* ) using that *π* *t* + 1 is greedy wrt. *v* *[π]* *[t]*

= max *a* *∈* *A* *[q]* *[π]* *[t]* [(] *[x]* [,] *[ a]* [)] using the definition of the Bellmanupdate (10.37)
*≥* *q* *[π]* *[t]* ( *x*, *π* *t* ( *x* ))

= *v* *[π]* *[t]* ( *x* )

= ˜ *v* 0 ( *x* ) . using (10.17)

Let us now consider a single iteration of fixed-point iteration. We have,

˜
###### v t + 1 ( x ) = r ( x, π t + 1 ( x )) + γ ∑ p ( x [′] | x, π t + 1 ( x )) · ˜ v t ( x [′] ) . using the definition of ˜ v t + 1 (10.21)

*x* *[′]* *∈* *X*

190 probabilistic artificial intelligence

Using an induction on *t*, we conclude,
###### ≥ r ( x, π t + 1 ( x )) + γ ∑ p ( x [′] | x, π t + 1 ( x )) · ˜ v t − 1 ( x [′] ) using the induction hypothesis,

*x* *[′]* *∈* *X* *v* ˜ *t* ( *x* *[′]* ) *≥* *v* ˜ *t* *−* 1 ( *x* *[′]* )
= ˜ *v* *t* ( *x* ) .

This establishes the first claim,

***v*** *[π]* *[t]* [+] [1] = lim (10.38)
*t* *→* ∞ ***[v]*** [˜] *[t]* *[ ≥]* ***[v]*** [˜] [0] [ =] ***[ v]*** *[π]* *[t]* [.]

For the second claim, recall from Bellman’s theorem (10.33) that *v* *[⋆]* is a
(unique) fixed-point of the Bellman update ***B*** *[⋆]* . [5] In particular, we have 5 We will show in eq. (10.39) that ***B*** *⋆* is
*v* *[π]* *[t]* [+] [1] *≡* *v* *[π]* *[t]* if and only if *v* *[π]* *[t]* [+] [1] *≡* *v* *[π]* *[t]* *≡* *v* *[⋆]* . In other words, if *v* *[π]* *[t]* *̸≡* *v* *[⋆]* a contraction, implying that *v* *[⋆]* is the
*unique* fixed-point of ***B*** *[⋆]* .

then eq. (10.38) is strict for at least one *x* *∈* *X* and *v* *[π]* *[t]* [+] [1] *̸≡* *v* *[π]* *[t]* . This
proves the strict monotonic improvement of policy iteration.

**Theorem 10.19** (Convergence of policy iteration) **.** *For finite Markov de-*
*cision processes, policy iteration converges to an optimal policy.*

*Proof.* Finite Markov decision processes only have a finite number of
deterministic policies (albeit exponentially many). Observe that policy
iteration only considers deterministic policies, and recall that there is
an optimal policy that is deterministic. As the value of policies strictly
increase in each iteration until an optimal policy is found, policy iteration must converge in finite time.

It can be shown that policy iteration converges to an exact solution in
a polynomial number of iterations. [6] Each iteration of policy iteration 6 Yinyu Ye. The simplex and policy
requires computing the value function, which we have seen to be of iteration methods are strongly polyno
mial for the markov decision problem

cubic complexity in the number of states. with a fixed discount rate. *Mathematics of*

*Operations Research*, 36(4):593–603, 2011

*10.3.4* *Value Iteration*

As we have mentioned, another natural approach of finding the optimal policy is to interpret *v* *[⋆]* as the fixed point of the Bellman update.
Recall our definition of the Bellman update from eq. (10.37),

( ***B*** *[⋆]* ***v*** )( *x* ) = max *a* *∈* *A* *[q]* [(] *[x]* [,] *[ a]* [)] [,]

where *q* was the state-action value function associated with the state
value function ***v*** . The value iteration algorithm is shown in alg. 10.20.

We will now prove the convergence of value iteration using the fixedpoint interpretation.

**Theorem 10.21** (Convergence of value iteration) **.** *Value iteration con-*
*verges to an optimal policy.*

markov decision processes 191

**Al** **g** **orithm 10.20:** Value iteration

**1** initialize *v* ( *x* ) *←* max *a* *∈* *A* *r* ( *x*, *a* ) for each *x* *∈* *X*

**2** `for` *t* = 1 `to` ∞ `do`

**3** *v* ( *x* ) *←* ( ***B*** *[⋆]* ***v*** )( *x* ) = max *a* *∈* *A* *q* ( *x*, *a* ) for each *x* *∈* *X*

**4** choose *π* *v*

*Proof.* Clearly, value iteration converges if *v* *[⋆]* is the unique fixed-point
of ***B*** *[⋆]* . We already know from Bellman’s theorem (10.33) that *v* *[⋆]* is
a fixed-point of ***B*** *[⋆]* . It remains to show that it is indeed the unique
fixed-point.

Analogously to our proof of the convergence of fixed-point iteration to
the value function *v* *[π]*, we show that ***B*** *[⋆]* is a contraction. Fix arbitrary
***v***, ***v*** *[′]* *∈* **R** *[n]*, then

*⋆* *⋆* *′* *⋆* *⋆* *′*
�� ***B*** ***v*** *−* ***B*** ***v*** �� ∞ [=] [ max] �� ( ***B*** ***v*** )( *x* ) *−* ( ***B*** ***v*** )( *x* ) �� using the definition of the *ℓ* ∞ norm
*x* *∈* *X* (10.23)


= max max using the definition of the Bellman
*x* *∈* *X* ���� *a* *∈* *A* *[q]* [(] *[x]* [,] *[ a]* [)] *[ −]* [max] *a* *∈* *A* *[q]* *[′]* [(] *[x]* [,] *[ a]* [)] ���� update (10.37)

*′*

*≤* max �� *q* ( *x*, *a* ) *−* *q* ( *x*, *a* ) �� using *|* max *x* *f* ( *x* ) *−* max *x* *g* ( *x* ) *| ≤*
*x* *∈* *X* [max] *a* *∈* *A*


= max
*x* *∈* *X*


*′*

*≤* max �� *q* ( *x*, *a* ) *−* *q* ( *x*, *a* ) �� using *|* max *x* *f* ( *x* ) *−* max *x* *g* ( *x* ) *| ≤*
*x* *∈* *X* [max] *a* *∈* *A*

max *x* *|* *f* ( *x* ) *−* *g* ( *x* ) *|*
*′* *′* *′*
*≤* *γ* max *p* ( *x* *[′]* *|* *x*, *a* ) �� ***v*** ( *x* ) *−* ***v*** ( *x* ) �� using the definition of the Q-function
###### x ∈ X [max] a ∈ A x [∑] [′] ∈ X (10.9) and | ∑ i a i | ≤ ∑ i | a i |

*′*
*≤* *γ* �� ***v*** *−* ***v*** �� ∞ (10.39) using ∑ *x* *′* *∈* *X* *p* ( *x* *[′]* *|* *x*, *a* ) = 1 and
*|* ***v*** ( *x* *[′]* ) *−* ***v*** *[′]* ( *x* *[′]* ) *| ≤∥* ***v*** *−* ***v*** *[′]* *∥* ∞


where *q* and *q* *[′]* are the state-action value functions associated with ***v***
and ***v*** *[′]*, respectively. By eq. (10.22), ***B*** *[⋆]* is a contraction and by Banach’s
fixed-point theorem *v* *[⋆]* is its unique fixed-point.



Value iteration converges to an *ϵ* -optimal solution in a polynomial
number of iterations. Unlike policy iteration, value iteration does not
converge to an exact solution in general. Recalling the update rule of
value iteration, its main benefit is that each iteration only requires a
sum over all possible actions *a* in state *x* and a sum over all reachable

192 probabilistic artificial intelligence

states *x* *[′]* from *x* . In sparse Markov decision processes, [7] an iteration of 7 Sparsity refers to the interconnectivity

value iteration can be performed in (virtually) constant time. of the state space. When only few states

are reachable from any state, we call an
MDP sparse.



markov decision processes 193



*10.4* *Partial Observability*

So far we have focused on the fully observable setting. That is, at
any time, our agent knows its current state. We have seen that we
can efficiently find the optimal policy (as long as the Markov decision
process is finite).


We have already encountered the partially observable setting in chapter 3, where we discussed Bayesian filtering and Kalman filters. In this
section, we consider how Markov decision processes can be extended ~~*A*~~ ~~1~~ ~~*A*~~ ~~2~~
to a partially observable setting where the agent can only access noisy
observations *Y* *t* of its state *X* *t* . *X* 1 *X* 2 *X* 3 *· · ·*




**Definition 10.25** (Partially observable Markov decision process, POMDP) **.**
Similarly to a Markov decision process, a *partially observable Markov de-*
*cision process* is specified by

- a set of *states X*,

- a set of *actions A*,

- *transition probabilities p* ( *x* *[′]* *|* *x*, *a* ), and

- a *reward function r* : *X* *×* *A* *→* **R** .

Additionally, it is specified by

- a set of *observations Y*, and

- *observation probabilities*

*o* ( *y* *|* *x* ) = . **P** ( *Y* *t* = *y* *|* *X* *t* = *x* ) . (10.42)

Whereas MDPs are controlled Markov chains, POMDPs are controlled

hidden Markov models.


Figure 10.4: Directed graphical model of
a partially observable Markov decision
process with hidden states *X* *t*, observables *Y* *t*, and actions *A* *t* .

Figure 10.5: Directed graphical model
of a hidden Markov model with hidden

states *X* *t* and observables *Y* *t* .




194 probabilistic artificial intelligence



such as speech recognition, decoding





POMDPs are a very powerful model, but very hard to solve in general. POMDPs can be reduced to a Markov decision process with an
enlarged state space. The key insight is to consider an MDP whose
states are the *beliefs*,

*b* *t* ( *x* ) = . **P** ( *X* *t* = *x* *|* *y* 1: *t*, *a* 1: *t* *−* 1 ), (10.45)

about the current state in the POMDP. In other words, the states of the

MDP are probability distributions over the states of the POMDP. We
will make this more precise in the following.

Let us assume that our prior belief about the state of our agent is
given by *b* 0 ( *x* ) = . **P** ( *X* 0 = *x* ) . Keeping track of how beliefs change
over time is known as *Bayesian filtering*, which we already encountered
in section 3.1. Given a prior belief *b* *t*, an action taken *a* *t*, and a new
observation *y* *t* + 1, the belief state can be updated as follows,

*b* *t* + 1 ( *x* ) = **P** ( *X* *t* + 1 = *x* *|* *y* 1: *t* + 1, *a* 1: *t* ) by the definition of beliefs (10.45)

= [1] using Bayes’ rule (1.58)

*Z* **[P]** [(] *[y]* *[t]* [+] [1] *[ |]* *[ X]* *[t]* [+] [1] [ =] *[ x]* [)] **[P]** [(] *[X]* *[t]* [+] [1] [ =] *[ x]* *[ |]* *[ y]* [1:] *[t]* [,] *[ a]* [1:] *[t]* [)]

= [1] using the definition of observation

*Z* *[o]* [(] *[y]* *[t]* [+] [1] *[ |]* *[ x]* [)] **[P]** [(] *[X]* *[t]* [+] [1] [ =] *[ x]* *[ |]* *[ y]* [1:] *[t]* [,] *[ a]* [1:] *[t]* [)]
probabilities (10.42)

markov decision processes 195

= [1] *p* ( *x* *|* *x* *[′]*, *a* *t* ) **P** � *X* *t* = *x* *[′]* *|* *y* 1: *t*, *a* 1: *t* *−* 1 � by conditioning on the previous state *x* *[′]*,
###### Z [o] [(] [y] [t] [+] [1] [ |] [ x] [)] [ ∑]

*x* *[′]* *∈* *X* noting *a* *t* does not influence *X* *t*


= [1] *p* ( *x* *|* *x* *[′]*, *a* *t* ) *b* *t* ( *x* *[′]* ) (10.46) using the definition of beliefs (10.45)
###### Z [o] [(] [y] [t] [+] [1] [ |] [ x] [)] [ ∑]

*x* *[′]* *∈* *X*


where
###### Z = . ∑ o ( y t + 1 | x ) ∑ p ( x | x [′], a t ) b t ( x [′] ) . (10.47)

*x* *∈* *X* *x* *[′]* *∈* *X*

Thus, the updated belief state is a deterministic mapping from the
previous belief state depending only on the (random) observation *y* *t* + 1
and the taken action *a* *t* . Note that this obeys a Markovian structure of
transition probabilities with respect to the beliefs *b* *t* .

The sequence of belief-states defines the sequence of random variables
( *B* *t* ) *t* *∈* **N** 0,

*B* *t* = . *X* *t* *|* *y* 1: *t*, *a* 1: *t* *−* 1, (10.48)

where the (state-)space of all beliefs is the (infinite) space of all probability distributions over *X*, [9] 9 This definition naturally extends to
continuous state spaces *X* .

*B* = . ∆ *X* = . � ***b*** *∈* **R** *[|]* *[X]* *[|]* : ***b*** *≥* **0**, ∑ *i* *[|]* = *[X]* *[|]* 1 ***[b]*** [(] *[i]* [) =] [ 1] �. (10.49)

A Markov decision process, where every belief corresponds to a state

is called a belief-state MDP.

**Definition 10.28** (Belief-state Markov decision process) **.** Given a POMDP,
the corresponding *belief-state Markov decision process* is a Markov decision process specified by

- the *belief space* *B* = . ∆ *X* depending on the *hidden states X*,

- the set of *actions A*,

- *transition probabilities*

*τ* ( *b* *[′]* *|* *b*, *a* ) = . **P** � *B* *t* + 1 = *b* *[′]* *|* *B* *t* = *b*, *A* *t* = *a* �, (10.50)

- and *rewards*
###### ρ ( b, a ) = . E x ∼ b [ r ( x, a )] = ∑ b ( x ) r ( x, a ) . (10.51)

*x* *∈* *X*

It remains to derive the transition probabilities *τ* in terms of the original POMDP. We have,

*τ* ( *b* *t* + 1 *|* *b* *t*, *a* *t* ) = **P** ( *b* *t* + 1 *|* *b* *t*, *a* *t* )
###### = ∑ P ( b t + 1 | b t, a t, y t + 1 ) P ( y t + 1 | b t, a t ) . (10.52) by conditioning on y t + 1 ∈ Y

*y* *t* + 1 *∈* *Y*

196 probabilistic artificial intelligence

Using the Markovian structure of the belief updates, we naturally set,


1 if *b* *t* + 1 matches the belief update of
eq. (10.45) given *b* *t*, *a* *t*, and *y* *t* + 1,

0 otherwise.

(10.53)


**P** ( *b* *t* + 1 *|* *b* *t*, *a* *t*, *y* *t* + 1 ) = .





The final missing piece is the likelihood of an observation *y* *t* + 1 given
the prior belief *b* *t* and action *a* *t*, which using our interpretation of beliefs corresponds to

**P** ( *y* *t* + 1 *|* *b* *t*, *a* *t* ) = **E** *x* *∼* *b* *t* � **E** *x* *′* *|* *x*, *a* *t* � **P** � *y* *t* + 1 *|* *X* *t* + 1 = *x* *[′]* [���]

= **E** *x* *∼* *b* *t* � **E** *x* *′* *|* *x*, *a* *t* � *o* ( *y* *t* + 1 *|* *x* *[′]* ) � [�] using the definition of observation
probabilities (10.42)
###### = ∑ b t ( x ) ∑ p ( x [′] | x, a t ) · o ( y t + 1 | x [′] ) . (10.54)

*x* *∈* *X* *x* *[′]* *∈* *X*


In principle, we can now apply arbitrary algorithms for planning in
MDPs to POMDPs. Of course, the problem is that there are infinitely
many beliefs, even for a finite state space *X* . [10] The belief-state MDP 10 You can think of an *|* *X* *|* -dimensional

has therefore an infinitely large belief space *B* . Even when only plan- space. Here, all points whose coordi
nates sum to 1 correspond to probability

ning over finite horizons, exponentially many beliefs can be reached. distributions (i.e., beliefs) over the hidSo the belief space blows-up very quickly. den states *X* . The convex hull of these

points is also known as the ( *|* *X* *| −* 1 )                                            
We will study MDPs with large state spaces (where transition dynam- dimensional *probability simplex* (cf. re
mark 1.10). Now, by definition of the

ics and rewards are unknown) in chapters 12 and 13. Similar methods ( *|* *X* *| −* 1 ) -dimensional probability simcan also be used to approximately solve POMDPs. plex as a polytope in *|* *X* *| −* 1 dimen
sions, we can conclude that its bound
A key idea in approximate solutions to POMDPs is that most belief ary consists of infinitely many points in

*|* *X* *|* dimensions. Noting that these points

states are never reached. A common approach is to discretize the belief corresponded to the probability distribuspace by sampling or by applying a dimensionality reduction. Exam- tions on *|* *X* *|*, we conclude that there are

infinitely many such distributions.


has therefore an infinitely large belief space *B* . Even when only planning over finite horizons, exponentially many beliefs can be reached.
So the belief space blows-up very quickly.


We will study MDPs with large state spaces (where transition dynamics and rewards are unknown) in chapters 12 and 13. Similar methods
can also be used to approximately solve POMDPs.


A key idea in approximate solutions to POMDPs is that most belief ary consists of infinitely many points in

*|* *X* *|* dimensions. Noting that these points

states are never reached. A common approach is to discretize the belief corresponded to the probability distribuspace by sampling or by applying a dimensionality reduction. Exam- tions on *|* *X* *|*, we conclude that there are

infinitely many such distributions.

ples are *point-based value iteration* (PBVI) and *point-based policy iteration*
(PBPI). [11] 11 Guy Shani, Joelle Pineau, and Robert
Kaplow. A survey of point-based pomdp
solvers. *Autonomous Agents and Multi-*



Even though we focus on the fully observed setting throughout this
course, the partially observed setting can be reduced to the fully observed setting with very large state spaces. In the next chapter, we
will consider learning and planning in unknown Markov decision processes (i.e., reinforcement learning) for small state spaces. The setting
of small state and action spaces is also known as the *tabular setting* .

markov decision processes 197


Then, in the final two chapters, we will consider approximate methods for large state and action spaces. In particular, in section 13.1, we
will revisit the problem of probabilistic planning in known Markov
decision processes, but with continuous state and action spaces.

### *11* *Tabular Reinforcement Learning*

*11.1*
*The Reinforcement Learning Problem*

Reinforcement learning is concerned with probabilistic planning in
unknown environments. This extends our study of known environments in the previous chapter. Those environments are still modeled
by Markov decision processes, but in reinforcement learning, we do
not know the dynamics *p* and rewards *r* in advance. Hence, reinforcement learning is at the intersection of the theories of probabilistic planning (i.e., Markov decision processes) and learning (e.g., multi-armed
bandits), which we covered extensively in the previous chapters.

We will continue to focus on the fully observed setting, where the
agent knows its current state. As we have seen in the previous section,
the partially observed setting corresponds to a fully observed setting
with very large state spaces. In this chapter, we will begin by considering reinforcement learning with small state and action spaces. This
setting is often called the *tabular setting*, as the value functions can be
computed exhaustively for all states and stored in a table.

Clearly, the agent needs to trade exploring and learning about the environment with exploiting its knowledge to maximize rewards. Thus,
the exploration-exploitation dilemma, which was at the core of Bayesian
optimization (see section 9.1), also plays a crucial role in reinforcement
learning. In fact, Bayesian optimization can be viewed as reinforcement learning with a fixed state and a continuous action space: In each
round, the agent plays an action, aiming to find the action that maximizes the reward. However, playing the same action multiple times
yields the same reward, implying that we remain in a single state. In
the context of Bayesian optimization, we used “regret” as performance
metric: in the jargon of planning, minimizing regret corresponds to
maximizing the cumulative reward.


*x* *t* + 1 *r* *t* *a* *t*

Figure 11.1: In reinforcement learning,
an agent interacts with its environment
in a sequence of rounds. After playing
an action *a* *t*, it observes rewards *r* *t* and
its new state *x* *t* + 1 . The agent then uses
this information to learn a model of the

world.

200 probabilistic artificial intelligence

Another key challenge of reinforcement learning is that the observed
data is dependent on the played actions. This is in contrast to the
setting of supervised learning that we have been considering in earlier
chapters, where the data is sampled independently.

*11.1.1* *Trajectories*

The data that the agent collects is modeled using so-called trajectories.

**Definition 11.1** (Trajectory) **.** A *trajectory τ* is a (possibly infinite) se
quence,

*τ* = ( . *τ* 0, *τ* 1, *τ* 2, . . . ), (11.1)

of *transitions*,

*τ* *i* = ( . *x* *i*, *a* *i*, *r* *i*, *x* *i* + 1 ), (11.2)

where *x* *i* *∈* *X* is the starting state, *a* *i* *∈* *A* is the played action, *r* *i* *∈* **R** is
the attained reward, and *x* *i* + 1 *∈* *X* is the ending state.

In the context of learning a dynamics and rewards model, *x* *i* and *a* *i* can
be understood as inputs, and *r* *i* and *x* *i* + 1 can be understood as labels
of a regression problem.


Crucially, the newly observed states *x* *t* + 1 and the rewards *r* *t* (across
multiple transitions) are conditionally independent given the previous
states *x* *t* and actions *a* *t* . This follows directly from the Markovian
structure of the underlying Markov decision process. [1] Formally, we 1 Recall the Markov property (6.6), which

have, assumes that in the underlying Markov

decision process (i.e., in our environment) the future state of an agent is inde
*X* *t* + 1 *⊥* *X* *t* *′* + 1 *|* *X* *t*, *X* *t* *′*, *A* *t*, *A* *t* *′*, (11.3a) pendent of past states given the agent’s

*R* *t* *⊥* *R* *t* *′* *|* *X* *t*, *X* *t* *′*, *A* *t*, *A* *t* *′*, (11.3b) current state. This is commonly called aMarkovian structure. From this Marko
vian structure, we gather that repeated

for any *t*, *t* *[′]* *∈* **N** 0 . In particular, if *x* *t* = *x* *t* *′* and *a* *t* = *a* *t* *′*, then *x* *t* + 1 encounters of state-action pairs result
and *x* *t* *′* + 1 are independent samples according to the transition model in independent trials of the transition

model and rewards.


have,


*X* *t* + 1 *⊥* *X* *t* *′* + 1 *|* *X* *t*, *X* *t* *′*, *A* *t*, *A* *t* *′*, (11.3a)


*R* *t* *⊥* *R* *t* *′* *|* *X* *t*, *X* *t* *′*, *A* *t*, *A* *t* *′*, (11.3b)


for any *t*, *t* *[′]* *∈* **N** 0 . In particular, if *x* *t* = *x* *t* *′* and *a* *t* = *a* *t* *′*, then *x* *t* + 1
and *x* *t* *′* + 1 are independent samples according to the transition model
*p* ( *X* *t* + 1 *|* *x* *t*, *a* *t* ) . Analogously, if *x* *t* = *x* *t* *′* and *a* *t* = *a* *t* *′*, then *r* *t* and
*r* *t* *′* are independent samples of the reward model *r* ( *x* *t*, *a* *t* ) . As we will
see later in this chapter and especially in chapter 13, this independence property is crucial for being able to learn about the underlying
Markov decision process. Notably, this implies that we can apply the
law of large numbers (1.83) and Hoeffding’s inequality (1.87) to our
estimators of both quantities.


The collection of data is commonly classified into two settings. In the
*episodic setting*, the agent performs a sequence of “training” rounds
(called *episodes* ). In the beginning of each round, the agent is reset to
some initial state. In contrast, in the *continuous setting* (or non-episodic,

tabular reinforcement learning 201

or online setting), the agent learns online. In particular, every action,
every reward, and every state transition counts.

The episodic setting is more applicable to an agent playing a computer
game. That is, the agent is performing in a simulated environment
that is easy to reset. The continuous setting is akin to an agent that
is deployed to the “real world”. In principle, real-world agents can
be trained in simulated environments before being deployed. However, this bears the risk of learning to exploit or rely on features of the
simulated environment that are not present in the real environment.
Sometimes, using a simulated environment for training is downright
impossible, as the real environment is too complex.

*11.1.2* *Control*

Another important distinction in how data is collected, is the distinction between on-policy and off-policy control. As the names suggest,
*on-policy methods* are used when the agent has control over its own actions, in other words, the agent can freely choose to follow any policy.
Being able to follow a policy is helpful, for example because it allows
the agent to experiment with trading exploration and exploitation.

In contrast, *off-policy methods* can be used even when the agent cannot
freely choose its actions. Off-policy methods are therefore able to make
use of observational data. This might be data that was collected by
another agent, a fixed policy, or during a previous episode. Off-policy
methods are therefore more *sample-efficient* than on-policy methods.
This is crucial, especially in settings where conducting experiments
(i.e., collecting new data) is expensive.

*11.2*
*Model-based Approaches*

Approaches to reinforcement learning are largely categorized into two
classes. *Model-based* approaches aim to learn the underlying Markov
decision process. More concretely, they learn models of the dynamics
*p* and rewards *r* . They then use these models to perform planning
(i.e., policy optimization) in the underlying Markov decision process.
In contrast, *model-free* approaches learn the value function directly. We
begin by discussing model-based approaches to the tabular setting. In
section 11.3, we cover model-free approaches.

*11.2.1* *Learning the Underlying Markov Decision Process*

Recall that the underlying Markov decision process was specified by
its dynamics *p* ( *x* *[′]* *|* *x*, *a* ) that correspond to the probability of entering

202 probabilistic artificial intelligence

state *x* *[′]* *∈* *X* when playing action *a* *∈* *A* from state *x* *∈* *X*, and its
rewards *r* ( *x*, *a* ) for playing action *a* *∈* *A* in state *x* *∈* *X* . A natural first
idea is to use maximum likelihood estimation to approximate these
quantities.

We can think of each transition *x* *[′]* *|* *x*, *a* as sampling from a categorical
random variable of which we want to estimate the success probabilities
for landing in each of the states. Therefore, as we have seen in example 1.50, the MLE of the dynamics model coincides with the sample

mean,

ˆ *[|]* *[ x]* [,] *[ a]* [)]
*p* ( *x* *[′]* *|* *x*, *a* ) = *[N]* [(] *[x]* *[′]* (11.4)

*N* ( *a* *|* *x* )

where *N* ( *x* *[′]* *|* *x*, *a* ) counts the number of transitions from state *x* to
state *x* *[′]* when playing action *a* and *N* ( *a* *|* *x* ) counts the number of
transitions that start in state *x* and play action *a* (regardless of the
next state). Similarly, for the rewards model, we obtain the following
maximum likelihood estimate (i.e., sample mean),


*r* ˆ ( *x*, *a* ) = 1
*N* ( *a* *|* *x* )


∞
###### ∑

*t* = 0
*x* *t* = *x*
*a* *t* = *a*


*r* *t* . (11.5)


It is immediate that both estimates are unbiased as both correspond to
a sample mean.

Still, for the models of our environment to become accurate, our agent
needs to visit *each* state-action pair ( *x*, *a* ) numerous times. Note that
our estimators for dynamics and rewards are only well-defined when
we visit the corresponding state-action pair at least once. However,
in a stochastic environment, a single visit will likely not result in an
accurate model. Hoeffding’s inequality (1.87) can be used to gauge
how accurate the estimates are after only a limited number of visits.

The next natural question is therefore how to use our current model of
the environment to pick actions such that exploration and exploitation
are traded effectively. This is what we will consider next.

*11.2.2* *ϵ-greedy Algorithm*

First, consider the following two extremes. If we always pick a random
action, we will eventually(!) estimate the dynamics and rewards correctly, yet we will do extremely poorly in terms of maximizing rewards
along the way. If we, on the other hand, always pick the best action
according to our current model, we will quickly get some reward, yet
we will likely get stuck in a suboptimal state. To trade exploration and
exploitation, a natural idea is to balance these two extremes.

tabular reinforcement learning 203

Arguably, the simplest idea is the following: At each time step, throw
a biased coin. If this coin lands heads, we pick an action uniformly at
random among all actions. If the coin lands tails, we pick the best action under our current model. This algorithm is called *ϵ-greedy*, where
the probability of a coin landing heads at time *t* is *ϵ* *t* .

**Al** **g** **orithm 11.2:** *ϵ*  - g reed y

**1** `for` *t* = 0 `to` ∞ `do`

**2** sample *u* *∈* Unif ([ 0, 1 ])

**3** `if` *u* *≤* *ϵ* *t* `then` pick action uniformly at random among all actions

**4** `else` pick best action under the current model

The *ϵ* -greedy algorithm provides a general framework for handling
the exploration-exploitation dilemma. Within this framework, how

the “best action” under the “current model” is chosen can be varied.

When the underlying MDP is learned using Monte Carlo estimation
as we discussed in section 11.2.1, the resulting algorithm is known as
*Monte Carlo control* . In this case, the best action under the model at time
*t* when the agent is at state *x* is the greedy action, arg max *a* *∈* *A* *Q* *t* ( *x*, *a* ),
where we denote by *Q* *t* ( *x*, *a* ) the estimated state-action value function
at time *t* with respect to ˆ *p* and ˆ *r* . However, the same framework can
also be used in the model-free setting where we pick the best action
without estimating the full underlying MDP. We discuss this approach
in greater detail in section 11.3.

Amazingly, this simple algorithm already works quite well. Nevertheless, it can clearly be improved. The key problem of *ϵ* -greedy is that it
explores the state space in an uninformed manner. In other words, it
explores ignoring all past experience. It thus does not eliminate clearly
suboptimal actions. This is a problem, especially as we typically have
many state-action pairs and when we recall that we have to explore
each such pair many times to learn an accurate model.


204 probabilistic artificial intelligence


is a fundamental requirement. There

*all* states when it only sees some state
not at all.





*11.2.3* *R* max *Algorithm*


Recall from our discussion of multi-armed bandits in section 9.2.1 that
a key principle in effectively trading exploration and exploitation is
“optimism in the face of uncertainty”. Let us apply this principle to
the reinforcement learning setting. The key idea is to assume that
the dynamics and rewards model “work in our favor” until we have
learned “good estimates” of the true dynamics and rewards.

More formally, if *r* ( *x*, *a* ) is unknown, we set ˆ *r* ( *x*, *a* ) = *R* max, where *R* max
is the maximum reward our agent can attain during a single transition.
Similarly, if *p* ( *x* *[′]* *|* *x*, *a* ) is unknown, we set ˆ *p* ( *x* *[⋆]* *|* *x*, *a* ) = 1, where *x* *[⋆]* is
a “fairy-tale state”. The fairy-tale state corresponds to everything our
agent could wish for, that is,

*p* ˆ ( *x* *[⋆]* *|* *x* *[⋆]*, *a* ) = 1 *∀* *a* *∈* *A*, (11.8)

*r* ˆ ( *x* *[⋆]*, *a* ) = *R* max *∀* *a* *∈* *A* . (11.9)

In practice, the decision of when to assume that the learned dynamics
and reward models are “good enough” has to be tuned.

In using these optimistic estimates of *p* and *r*, we obtain an optimistic
underlying Markov decision process that exhibits a bias towards exploration. In particular, the rewards attained in this MDP, are an upper



Figure 11.2: Illustration of the fairy-tale
state of *R* max . If in doubt, the agent believes actions from the state *x* to lead to
the fairy-tale state *x* *[⋆]* with maximal rewards. This encourages the exploration
of unknown states.

tabular reinforcement learning 205

bound of the true reward. The resulting algorithm is known as the
*R* max *algorithm* .

**Al** **g** **orithm 11.6:** *R* max al g orithm

**1** add the fairy-tale state *x* *[⋆]* to the Markov decision process

**2** set ˆ *r* ( *x*, *a* ) = *R* max for all *x* *∈* *X* and *a* *∈* *A*

**3** set ˆ *p* ( *x* *[⋆]* *|* *x*, *a* ) = 1 for all *x* *∈* *X* and *a* *∈* *A*

**4** compute the optimal policy ˆ *π* for ˆ *r* and ˆ *p*

**5** `for` *t* = 0 `to` ∞ `do`

**6** execute policy ˆ *π* (for some number of steps)

**7** for each visited state-action pair ( *x*, *a* ), update ˆ *r* ( *x*, *a* )

**8** estimate transition probabilities ˆ *p* ( *x* *[′]* *|* *x*, *a* )

**9** after observing “enough” transitions and rewards, recompute the
optimal policy ˆ *π* according the current model ˆ *p* and ˆ *r* .

How many transitions are “enough”? We can use Hoeffding’s inequality (1.87) to get a rough idea! The key, here, is our observation from
eq. (11.3) that the transitions and rewards are conditionally independent given the state-action pairs. As we have discussed in section 6.1.4
on the ergodic theorem, Hoeffding’s inequality does not hold for dependent samples. In particular, Hoeffding’s inequality tells us that
for the absolute approximation error to be below *ϵ* with probability at
least 1 *−* *δ*, we need

max
*N* ( *a* *|* *x* ) *≥* *[R]* [2] (11.10) see (1.88)
2 *ϵ* [2] [ log 2] *δ* [.]

**Lemma 11.7** (Exploration and exploitation of *R* max ) **.** *Every T time steps,*
*with high probability, R* max *either*

 - *obtains near-optimal reward; or*

 - *visits at least one unknown state-action pair.* [4] 4 Note that in the tabular setting, there
are “only” polynomially many state*Here, T depends on the mixing time of the Markov chain induced by the*
action pairs.
*optimal policy.*

**Theorem 11.8** (Convergence of *R* max ) **.** *With probability at least* 1 *−* *δ,*
*R* max *reaches an ϵ-optimal policy in a number of steps that is polynomial in*
*|* *X* *|* *,* *|* *A* *|* *, T,* [1] / *ϵ* *,* [1] / *δ* *, and R* max *.* [5] 5 Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial
time algorithm for near-optimal rein*11.2.4* *Challenges* forcement learning. *Journal of Machine*
*Learning Research*, 3(Oct):213–231, 2002

We have seen that the *R* max algorithm performs remarkably well in the
tabular setting. However, there are important computational limitations to the model-based approaches that we presented in this chapter.

206 probabilistic artificial intelligence

First, observe that the (tabular) model-based approach requires us to
store ˆ *p* ( *x* *[′]* *|* *x*, *a* ) and ˆ *r* ( *x*, *a* ) in a table. This table already has *O* � *n* [2] *m* �

entries. Even though polynomial in the size of the state and action
spaces, this quickly becomes unmanageable.

Second, the model-based approach requires us to “solve” the learned
Markov decision processes to obtain the optimal policy (using policy
or value iteration). As we continue to learn over time, we need to find
the optimal policy many times. *R* max recomputes the policy after each
state-action pair is observed sufficiently often, so *O* ( *nm* ) times.

*11.3* *Model-free Approaches*

In the previous section, we have seen that learning and remembering
the model as well as planning within the estimated model can potentially be quite expensive in the model-based approach. We therefore
turn to model-free methods that estimate the value function directly.
Thus, they require neither remembering the full model nor planning
in the underlying Markov decision process. We will, however, return
to model-based methods in chapter 13 to see that a lot of promise lies
in combining methods from model-based reinforcement learning with
methods from model-free reinforcement learning.

A significant benefit to model-based reinforcement learning is that it
is inherently off-policy. That is, any trajectory regardless of the policy
used to obtain it can be used to improve the model of the underlying
Markov decision process. In the model-free setting, this not necessarily true. By default, estimating the value function according to the
data from a trajectory, will yield an estimate of the value function corresponding to the policy that was used to sample the data.

We will start by discussing on-policy methods and later see how the
value function can be estimated off-policy.

*11.3.1* *On-policy Value Estimation*

Let us suppose, our agent follows a fixed policy *π* . Then, the corresponding value function *v* *[π]* is given as
###### v [π] ( x ) = r ( x, π ( x )) + γ ∑ p ( x [′] | x, π ( x )) · v [π] ( x [′] ) using the definition of the value

*x* *[′]* *∈* *X* function (10.7)


= **E** *R* 0, *X* 1 [ *R* 0 + *γv* *[π]* ( *X* 1 ) *|* *X* 0 = *x*, *A* 0 = *π* ( *x* )] (11.11) interpreting the above expression as an
expectation over the random quantities

Our first instinct might be to use a Monte Carlo estimate of this expec- *R* 0 and *X* 1


(11.11)


Our first instinct might be to use a Monte Carlo estimate of this expectation. Due to the conditional independence of the transitions (11.3),
Monte Carlo approximation does yield an unbiased estimate,


*≈* *r* + *γv* *[π]* ( *x* *[′]* ) (11.12)

tabular reinforcement learning 207

where the agent observed the transition ( *x*, *a*, *r*, *x* *[′]* ) . Note that to estimate this expectation we use a single(!) sample, [6] unlike our previ- 6 The idea is that we will use this ap
ous applications of Monte Carlo sampling where we usually averaged proximation repeatedly as our agent col
lects new data to achieve the same effect

over *m* samples. However, there is one significant problem in this ap- as initially averaging over multiple samproximation. Our approximation of *v* *[π]* does in turn depend on the ples.
(unknown) true value of *v* *[π]* !

The key idea is to use a bootstrapping estimate of the value function
instead. That is, in place of the true value function *v* *[π]*, we will use a
“running estimate” *V* *[π]* . In other words, whenever observing a new
transition, we use our previous best estimate of *v* *[π]* to obtain a new
estimate *V* *[π]* . We already encountered bootstrapping briefly in section 7.4.4 in the context of probabilistic ensembles in Bayesian deep
learning. More generally, *bootstrapping* refers to approximating a true
quantity (e.g., *v* *[π]* ) by using an empirical quantity (e.g., *V* *[π]* ), which itself is constructed using samples from the true quantity that is to be
approximated.

Due to its use in estimating the value function, bootstrapping is a core
concept to model-free reinforcement learning. Crucially, using a bootstrapping estimate generally results in biased estimates of the value
function. Moreover, due to relying on a single sample, the estimates
from eq. (11.12) tend to have very large variance.

The variance of the estimate is typically reduced by mixing new estimates of the value function with previous estimates using a learning
rate *α* *t* . This yields the *temporal-difference learning* algorithm.

**Al** **g** **orithm 11.9:** Tem p oral-difference (TD) learnin g

**1** initialize *V* *[π]* arbitrarily (e.g., as **0** )

**2** `for` *t* = 0 `to` ∞ `do`

**3** follow policy *π* to obtain the transition ( *x*, *a*, *r*, *x* *[′]* )

**4** *V* *[π]* ( *x* ) *←* ( 1 *−* *α* *t* ) *V* *[π]* ( *x* ) + *α* *t* ( *r* + *γV* *[π]* ( *x* *[′]* )) `//` (11.13)

The update rule is sometimes written equivalently as

*V* *[π]* ( *x* ) *←* *V* *[π]* ( *x* ) + *α* *t* ( *r* + *γV* *[π]* ( *x* *[′]* ) *−* *V* *[π]* ( *x* )) . (11.14)

Thus, the update to *V* *[π]* ( *x* ) is proportional to the learning rate and the
difference between the previous estimate and the renewed estimate
using the new observation.

**Theorem 11.10** (Convergence of TD-learning) **.** *If* ( *α* *t* ) *t* *∈* **N** 0 *satisfies the*
*RM-conditions* (1.108) *and all state-action pairs are chosen infinitely often,*
*then V* *[π]* *converges to v* *[π]* *with probability* 1 *.* [7] 7 Tommi Jaakkola, Michael Jordan, and
Satinder Singh. Convergence of stochastic iterative dynamic programming algorithms. *Advances in neural information*
*processing systems*, 6, 1993

208 probabilistic artificial intelligence

Importantly, note that due to the Monte Carlo approximation of eq. (11.11)
with respect to transitions attained by following policy *π*, TD-learning
is a fundamentally on-policy method. That is, for the estimates *V* *[π]* to
converge to the true value function *v* *[π]*, the transitions that are used for
the estimation must follow policy *π* .



expectation over *R* 0, *X* 1 and *A* 1

single sample



*Q* *[π]* ( *x* *[′]*, *a* *[′]* ) = *Q* *[π]* ( *x* *[′]*, *π* ( *x* *[′]* )) = *V* *[π]* ( *x* *[′]* ) if

policy *π* .



step on-policy reinforcement-learning
algorithms” (Singh et al.)
To find the optimal policy *π* *[⋆]*, we can use an analogue of policy iteration (see alg. 10.17). In each iteration *t*, we estimate the value function
*q* *[π]* *[t]* of policy *π* *t* with the estimate *Q* *[π]* *[t]* obtained from SARSA. We then
choose the greedy policy with respect to *Q* *[π]* *[t]* as the next policy *π* *t* + 1 .
However, due to the on-policy nature of SARSA, we cannot reuse any
data between the iterations. Moreover, it turns out that in practice,
when using only finitely many samples, this form of greedily optimizing Markov decision processes does not explore enough. At least
partially, this can be compensated for by injecting noise when choosing
the next action, e.g., by following an *ϵ* -greedy policy.

tabular reinforcement learning 209

*11.3.2* *Off-policy Value Estimation*

Consider the following slight adaptation of the derivation of SARSA
(11.16),
###### q [π] ( x, a ) = r ( x, a ) + γ ∑ p ( x [′] | x, a ) ∑ π ( a [′] | x [′] ) q [π] ( x [′], a [′] ) using Bellman’s expectation equation

*x* *[′]* *∈* *X* *a* *[′]* *∈* *A* (10.15)


= *a* interpreting the above expression as an

expectation over *R* 0 and *X* 1
(11.18)


�


= **E** *R* 0, *X* 1


�

###### R 0 + γ ∑ π ( a [′] | X 1 ) q [π] ( X 1, a [′] ) | X 0 = x, A 0 = a

*a* *[′]* *∈* *A*

###### ≈ r + γ ∑ π ( a [′] | x [′] ) q [π] ( x [′], a [′] ) (11.19) Monte Carlo approximation with a

*a* *[′]* *∈* *A* single sample

where the agent observed the transition ( *x*, *a*, *r*, *x* *[′]* ) . This yields the
update rule,


.

�


*Q* *[π]* ( *x*, *a* ) *←* ( 1 *−* *α* *t* ) *Q* *[π]* ( *x*, *a* ) + *α* *t*


�

###### r + γ ∑ π ( a [′] | x [′] ) Q [π] ( x [′], a [′] )

*a* *[′]* *∈* *A*


(11.20)

This adapted update rule *explicitly* chooses the subsequent action *a* *[′]* according to policy *π* whereas SARSA absorbs this choice into the Monte
Carlo approximation. The algorithm has analogous convergence guar
antees to those of SARSA.

Crucially, this algorithm is off-policy. That is, we can use transitions
that were obtained according to *any* policy to estimate the value of a
fixed policy *π*, which we may have never used! Perhaps this seems
contradictory at first, but it is not. As noted, the key difference to
the on-policy TD-learning and SARSA is that our estimate of the Qfunction explicitly keeps track of the next-performed action. It does so
for any action in any state. Moreover, note that the transitions that are
due to the dynamics model and rewards are unaffected by the used
policy. They merely depend on the originating state-action pair. We
can therefore use the instances where other policies played action *π* ( *x* )
in state *x* to estimate the performance of *π* .

*11.3.3* *Q-learning*

It turns out that there is a way to estimate the value function of the
optimal policy directly. Recall from Bellman’s theorem (10.32) that the
optimal policy *π* *[⋆]* can be characterized in terms of the optimal stateaction value function *q* *[⋆]*,

*π* *[⋆]* ( *x* ) = arg max *q* *[⋆]* ( *x*, *a* ) .
*a* *∈* *A*

*π* *[⋆]* corresponds to greedily maximizing the value function.

210 probabilistic artificial intelligence

Analogously to our derivation of SARSA (11.16), only using Bellman’s
theorem (10.33) in place of Bellman’s expectation equation (10.15), we
obtain,

###### q [⋆] ( x, a ) = r ( x, a ) + γ ∑ p ( x [′] | x, a ) max using that the Q-function is a

*x* *[′]* *∈* *X* *a* *[′]* *∈* *A* *[q]* *[⋆]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] fixed-point of the Bellman update, see
Bellman’s theorem (10.33)
= **E** *R* 0, *X* 1 *R* 0 + *γ* max (11.21) interpreting the above expression as an

*[q]* *[⋆]* [(] *[X]* [1] [,] *[ a]* *[′]* [)] *[ |]* *[ X]* [0] [ =] *[ x]* [,] *[ A]* [0] [ =] *[ a]*


� *R* 0 + *γ* max *a* *[′]* *∈* *A* *[q]* *[⋆]* [(] *[X]* [1] [,] *[ a]* *[′]* [)] *[ |]* *[ X]* [0] [ =] *[ x]* [,] *[ A]* [0] [ =] *[ a]* �


(11.21) interpreting the above expression as an
expectation over *R* 0 and *X* 1


*≈* *r* + *γ* max *a* *[′]* *∈* *A* *[q]* *[⋆]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] (11.22) Monte Carlo approximation with asingle sample

where the agent observed the transition ( *x*, *a*, *r*, *x* *[′]* ) . Using a bootstrapping estimate *Q* *[⋆]* for *q* *[⋆]*, we obtain a structurally similar algorithm to
TD-learning and SARSA — only for estimating the optimal Q-function
directly! This algorithm is known as *Q-learning* . Whereas we have seen
that the optimal policy can be found using TD-learning (or SARSA) in
a policy-iteration-like scheme, Q-learning is conceptually similar to

value iteration.

**Al** **g** **orithm 11.13:** Q-learnin g

**1** initialize *Q* *[⋆]* ( *x*, *a* ) arbitrarily (e.g., as **0** )

**2** `for` *t* = 0 `to` ∞ `do`

**3** observe the transition ( *x*, *a*, *r*, *x* *[′]* )

**4** *Q* *[⋆]* ( *x*, *a* ) *←* ( 1 *−* *α* *t* ) *Q* *[⋆]* ( *x*, *a* ) + *α* *t* ( *r* + *γ* max *a* *′* *∈* *A* *Q* *[⋆]* ( *x* *[′]*, *a* *[′]* ))

`//` (11.23)

Similarly to TD-learning, the update rule can also be expressed as


*Q* *[⋆]* ( *x*, *a* ) *←* *Q* *[⋆]* ( *x*, *a* ) + *α* *t*


� *r* + *γ* max *a* *[′]* *∈* *A* *[Q]* *[⋆]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] *[ −]* *[Q]* *[⋆]* [(] *[x]* [,] *[ a]* [)] �. (11.24)


Crucially, the Monte Carlo approximation of eq. (11.21) does not depend on the policy. Thus, Q-learning is an off-policy method.

**Theorem 11.14** (Convergence of Q-learning) **.** *If* ( *α* *t* ) *t* *∈* **N** 0 *satisfies the*
*RM-conditions* (1.108) *and all state-action pairs are chosen infinitely often*
*(that is, the sequence of policies used to obtain the transitions is GLIE), then*
*Q* *[⋆]* *converges to q* *[⋆]* *with probability* 1 *.* [10] 10 Tommi Jaakkola, Michael Jordan, and
Satinder Singh. Convergence of stochas
It can be shown that with probability at least 1 *−* *δ*, Q-learning con- tic iterative dynamic programming alverges to an *ϵ* -optimal policy in a number of steps that is polynomial gorithms. *processing systems Advances in neural information*, 6, 1993
in log *|* *X* *|*, log *|* *A* *|*, [1] / *ϵ* and log [1] / *δ* . [11] 11 Eyal Even-Dar, Yishay Mansour, and
Peter Bartlett. Learning rates for qlearning. *Journal of machine learning Re-*


tabular reinforcement learning 211




|0|0|A|B|
|---|---|---|---|
|+10|+1|G 1|G 2|




|Episode 1|Episode 2|
|---|---|
|Episode 1|x a x′ r|
|x a x′ r||
||B ← A 0 A ↓ G 0 1 G exit 10 1|
|A ↓ G 0 1 G exit 10 1||
|||



*11.3.4* *Optimistic Q-learning*

The next natural question is how to effectively trade exploration and
exploitation to both visit all state-action pairs many times, but also
attain a high reward.





212 probabilistic artificial intelligence



However, as we have seen in section 11.2.2, random “uninformed”

exploration like *ϵ* -greedy and softmax exploration explores the state
space very slowly. We therefore return to the principle of “optimism
in the face of uncertainty”, which already led us to the *R* max algorithm
in the model-based setting. We will now additionally assume that the
rewards are non-negative, that is, 0 *≤* *r* ( *x*, *a* ) *≤* *R* max ( *∀* *x* *∈* *X*, *a* *∈* *A* ) .
It turns out that a similar algorithm to *R* max also exists for (model-free)
Q-learning: it is called *optimistic Q-learning* and shown in alg. 11.17.

**Al** **g** **orithm 11.17:** O p timistic Q-learnin g

**1** initialize *Q* *[⋆]* ( *x*, *a* ) = *V* max ∏ *t* *[T]* = [init] 1 [(] [1] *[ −]* *[α]* *[t]* [)] *[−]* [1]

**2** `for` *t* = 0 `to` ∞ `do`

**3** pick action *a* *t* = arg max *a* *∈* *A* *Q* *[⋆]* ( *x*, *a* ) and observe the transition
( *x*, *a*, *r*, *x* *[′]* )

**4** *Q* *[⋆]* ( *x*, *a* ) *←* ( 1 *−* *α* *t* ) *Q* *[⋆]* ( *x*, *a* ) + *α* *t* ( *r* + *γ* max *a* *′* *∈* *A* *Q* *[⋆]* ( *x* *[′]*, *a* *[′]* ))

`//` (11.26)

Here,

. *R* max
*V* max = max

1 *−* *γ* *[≥]* *x* *∈* *X*, *a* *∈* *A* *[q]* *[⋆]* [(] *[x]* [,] *[ a]* [)]

is an upper bound on the discounted return and *T* init is some initialization time. Intuitively, the initialization of *Q* *[⋆]* corresponds to the
best-case long-term reward, assuming that all individual rewards are
upper bounded by *R* max . This is shown by the following lemma.

**Lemma 11.18.** *Denote by Q* *[⋆]* *t* *[, the approximation of q]* *[⋆]* *[attained in the t-th]*
*iteration of optimistic Q-learning. Then, for any state-action pair* ( *x*, *a* ) *and*
*iteration t such that N* ( *a* *|* *x* ) *≤* *T* init *,* [12] 12 *N* ( *a* *|* *x* ) is the number of times action
*a* is performed in state *x* .
*Q* *[⋆]* *t* [(] *[x]* [,] *[ a]* [)] *[ ≥]* *[V]* [max] *[≥]* *[q]* *[⋆]* [(] *[x]* [,] *[ a]* [)] [.] (11.27)

*Proof.* We write *β* *τ* = . ∏ *iτ* = 1 [(] [1] *[ −]* *[α]* *[i]* [)] [ and] *[ η]* *[i]* [,] *[τ]* = . *α* *i* ∏ *τj* = *i* + 1 [(] [1] *[ −]* *[α]* *[j]* [)] [. Using]
the update rule of optimistic Q-learning (11.26), we have


*Q* *[⋆]* *t* [(] *[x]* [,] *[ a]* [) =] *[ β]* *N* ( *a* *|* *x* ) *[Q]* 0 *[⋆]* [(] *[x]* [,] *[ a]* [) +]


*N* ( *a* *|* *x* )
###### i ∑ = 1 η i, N ( a | x ) ( r + γ max a i ∈ A [Q] t [⋆] i [(] [x] i [,] [ a] i [))]

(11.28)

tabular reinforcement learning 213

where *x* *i* is the next state arrived at time *t* *i* when action *a* is performed

the *i* -th time in state *x* .

Using the assumption that the rewards are non-negative, from eq. (11.28)
and *Q* 0 *[⋆]* [(] *[x]* [,] *[ a]* [) =] *[ V]* [max] [/] *[β]* *[T]* init [, we immediately have]

*β* *N* ( *a* *|* *x* )
*Q* *[⋆]* *t* [(] *[x]* [,] *[ a]* [)] *[ ≥]* *V* max

*β* *T* init

*≥* *V* max . using *N* ( *a* *|* *x* ) *≤* *T* init

Now, if *T* init is chosen large enough, it can be shown that optimistic
Q-learning converges quickly to an optimal policy.

**Theorem 11.19** (Convergence of optimistic Q-learning) **.** *With probabil-*
*ity at least* 1 *−* *δ, optimistic Q-learning obtains an ϵ-optimal policy after a*
*number of steps that is polynomial in* *|* *X* *|* *,* *|* *A* *|* *,* [1] / *ϵ* *,* log [1] / *δ* *, and R* max *where*
*the initialization time T* init *is upper bounded by a polynomial in the same*
*coefficients.* [13] 13 Eyal Even-Dar and Yishay Mansour.
Convergence of optimistic and increNote that for Q-learning, we still need to store *Q* *[⋆]* ( *x*, *a* ) for any state- mental q-learning. *Advances in neural in-*
action pair in memory. Thus, Q-learning requires *O* ( *nm* ) memory. *formation processing systems*, 14, 2001
During each transition, we need to compute

max *a* *∈* *A* *[Q]* *[⋆]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)]

once. If we run Q-learning for *T* iterations, this yields a time complexity of *O* ( *Tm* ) . Crucially, for sparse Markov decision processes where,
in most states, only few actions are permitted, each iteration of Qlearning can be performed in (virtually) constant time. This is a big
improvement of the quadratic (in the number of states) performance
of the model-based *R* max algorithm.

*11.3.5* *Challenges*

We have seen that both the model-based *R* max algorithm and the modelfree Q-learning take time polynomial in the number of states *|* *X* *|* and
the number of actions *|* *A* *|* to converge. While this is acceptable in small
grid worlds, this is completely unacceptable for large state and action

spaces.

Often, domains are continuous, for example when modeling beliefs
about states in a partially observable environment. Also, in many
structured domains (e.g., chess or multiagent planning), the size of the
state and action space is exponential in the size of the input. In the final
two chapters, we will therefore explore how model-free and modelbased methods can be used (approximately) in such large domains.

214 probabilistic artificial intelligence


### *12* *Model-free Approximate Reinforcement Learning*

In the previous chapter, we have seen methods for tabular settings.
Our goal now is to extend the model-free methods like TD-learning
and Q-learning to large state-action spaces *X* and *A* . We have seen
that a crucial bottleneck of these methods is the parameterization of

the value function. If we want to store the value function in a ta
ble, we need at least *O* ( *|X |* ) space. If we learn the Q function, we
even need *O* ( *|X | · |A|* ) space. Also, for large state-action spaces, the
time required to compute the value function for every state-action pair
exactly will grow polynomially in the size of the state-action space.
Hence, a natural idea is to learn *approximations* of these functions with
a low-dimensional parameterization. Such approximations were the
focus of the first few chapters and are, in fact, the key idea behind
methods for reinforcement learning in large state-action spaces.

*12.1*
*Tabular Reinforcement Learning as Optimization*

To begin with, let us reinterpret the model-free methods from the previous section, TD-learning and Q-learning, as solving an optimization
problem, where each iteration corresponds to a single gradient update.
We will focus on TD-learning here, but the same interpretation applies
to Q-learning. Recall the update rule of TD-learning (11.13),

*V* *[π]* ( *x* ) *←* ( 1 *−* *α* *t* ) *V* *[π]* ( *x* ) + *α* *t* ( *r* + *γV* *[π]* ( *x* *[′]* )) .

Note that this looks just like the update rule of an optimization algorithm! We can parameterize our estimates *V* *[π]* with parameters ***θ*** that
are updated according to the gradient of some loss function, assuming fixed bootstrapping estimates. In particular, in the tabular setting
(i.e., over a discrete domain), we can parameterize the value function
exactly by learning a separate parameter for each state,

***θ*** = [ [.] ***θ*** ( 1 ), . . ., ***θ*** ( *n* )], *V* *[π]* ( *x* ; ***θ*** ) = . ***θ*** ( *x* ) . (12.1)

216 probabilistic artificial intelligence

To re-derive the above update rule as a gradient update, let us consider
the following loss function,

*ℓ* ( ***θ*** ; *x*, *r* ) = . 1 (12.2)

2 [(] *[v]* *[π]* [(] *[x]* [)] *[ −]* ***[θ]*** [(] *[x]* [))] [2]


= [1]

2


2
� *r* + *γ* **E** *x* *′* *|* *x*, *π* ( *x* ) � *v* *[π]* ( *x* *[′]* ) � *−* ***θ*** ( *x* ) � (12.3) using Bellman’s expectation equation
(10.11)


Note that this loss corresponds to a standard squared loss of the difference between the parameter ***θ*** ( *x* ) and the label *v* *[π]* ( *x* ) we want to

learn.

We can now find the gradient of this loss. Elementary calculations
yield,

***∇*** ***θ*** ( *x* ) *ℓ* ( ***θ*** ; *x*, *r* ) = ***θ*** ( *x* ) *−* � *r* + *γ* **E** *x* *′* *|* *x*, *π* ( *x* ) � *v* *[π]* ( *x* *[′]* ) � [�] . (12.4)

Now, we cannot compute this derivative because we cannot compute
the expectation. Firstly, the expectation is over the true value function which is unknown to us. Secondly, the expectation is over the
transition model which we are trying to avoid in model-free methods.

To resolve the first issue, analogously to TD-learning, instead of learning the true value function *v* *[π]* which is unknown, we learn the bootstrapping estimate *V* *[π]* . Recall that the core principle behind bootstrapping as discussed in section 11.3.1 is that this bootstrapping estimate
*V* *[π]* is *treated* as if it were independent of the current estimate of the
value function ***θ*** . To emphasize this, we write *V* *[π]* ( *x* ; ***θ*** [old] ) *≈* *v* *[π]* ( *x* )
where ***θ*** [old] = ***θ*** but ***θ*** [old] is treated as a constant with respect to ***θ*** . [1] 1 That is, the bootstrapping estimate
*V* *[π]* ( *x* ; ***θ*** [old] ) is assumed to be constant

To resolve the second issue, analogously to the introduction of TD- with respect to ***θ*** ( *x* ) in the same way that

*v* *[π]* ( *x* ) is constant with respect to ***θ*** ( *x* ) .

learning in the previous chapter, we will use a Monte Carlo estimate If we were not using the bootstrapping
using a single sample. Recall that this is only possible because the estimate, the following derivation of the

gradient of the loss would not be this

transitions are conditionally independent given the state-action pair.

simple.



tic approaches to alleviate this problem



Using the aforementioned shortcuts, let us define the loss *ℓ* after ob
model-free approximate reinforcement learning 217


serving the single transition ( *x*, *a*, *r*, *x* *[′]* ),


*ℓ* ( ***θ*** ; *x*, *r*, *x* *[′]* ) = . 1

2


2
*r* + *γV* *[π]* ( *x* *[′]* ; ***θ*** [old] ) *−* ***θ*** ( *x* ) . (12.5)
� �


We define the gradient of this loss with respect to ***θ*** ( *x* ) as

. *′*
*δ* TD = ***∇*** ***θ*** ( *x* ) *ℓ* ( ***θ*** ; *x*, *r*, *x* )

= ***θ*** ( *x* ) *−* *r* + *γV* *[π]* ( *x* *[′]* ; ***θ*** [old] ) . (12.6)
� �


This error term is also called *temporal-difference (TD) error* . The temporal difference error compares the previous estimate of the value function to the bootstrapping estimate of the value function. We know
from the law of large numbers (1.83) that Monte Carlo averages are
unbiased. [3] We therefore have, 3 Crucially, the samples are unbiased
with respect to the approximate label in

**E** *x* *′* *|* *x*, *π* ( *x* ) [ *δ* TD ] = ***∇*** ***θ*** ( *x* ) *ℓ* ( ***θ*** ; *x*, *r* ) . (12.7) terms of the bootstrapping estimate only.

Due to bootstrapping the value function,
the estimates are not unbiased with re
Naturally, we can use these unbiased gradient estimates with respect spect to the true value function. More
over, the variance of each individual es
to the loss *ℓ* to perform stochastic gradient descent. This yields the

timation of the gradient is large, as we

update rule, only consider a single transition.


**E** *x* *′* *|* *x*, *π* ( *x* ) [ *δ* TD ] = ***∇*** ***θ*** ( *x* ) *ℓ* ( ***θ*** ; *x*, *r* ) . (12.7)


Naturally, we can use these unbiased gradient estimates with respect
to the loss *ℓ* to perform stochastic gradient descent. This yields the
update rule,


*V* *[π]* ( *x* ; ***θ*** ) = ***θ*** ( *x* ) *←* ***θ*** ( *x* ) *−* *α* *t* *δ* TD (12.8) using stochastic gradient descent with
learning rate *α* *t*, see alg. 1.61

= ( 1 *−* *α* *t* ) ***θ*** ( *x* ) + *α* *t* *r* + *γV* *[π]* ( *x* *[′]* ; ***θ*** [old] ) using the definition of the temporal
� �
difference error (12.6)
= ( 1 *−* *α* *t* ) *V* *[π]* ( *x* ; ***θ*** ) + *α* *t* *r* + *γV* *[π]* ( *x* *[′]* ; ***θ*** [old] ) . (12.9) substituting *V* *[π]* ( *x* ; ***θ*** ) for ***θ*** ( *x* )
� �

Observe that this gradient update coincides with the update rule of
TD-learning (11.13). Therefore, TD-learning is essentially performing
stochastic gradient descent using the TD-error as an unbiased gradient
estimate. [4] Crucially, TD-learning performs stochastic gradient descent 4 An alternative interpretation is that
with respect to the bootstrapping estimate of the value function *V* *[π]* TD-learning performs gradient descent
with respect to the loss *ℓ* .

and not the true value function *v* *[π]* ! Stochastic gradient descent with
a bootstrapping estimate is also called *stochastic semi-gradient descent* .
Importantly, the optimization target *r* + *γV* *[π]* ( *x* *[′]* ; ***θ*** [old] ) from the *ℓ* loss
is now *moving* between iterations. We have seen in the previous chapter that using a bootstrapping estimate still guarantees (asymptotic)
convergence to the true value function.

*12.2*
*Value Function Approximation*

To scale to large state spaces, it is natural to approximate the value
function using a parameterized model, *V* ( ***x*** ; ***θ*** ) or *Q* ( ***x***, ***a*** ; ***θ*** ) . You may
think of this as a regression problem where we map state(-action) pairs
to a real number. Recall from the previous section that this is a strict
generalization of the tabular setting, as we could use a separate pa
rameter to learn the value function for each individual state-action

218 probabilistic artificial intelligence

pair. Our goal for large state-action spaces is to exploit the smoothness properties [5] of the value function to condense the representation. 5 That is, the value function takes similar
values in “similar” states.
A simple idea is to use a linear function approximation with the handdesigned feature map ***ϕ***,

*Q* ( ***x***, ***a*** ; ***θ*** ) = . ***θ*** *⊤* ***ϕ*** ( ***x***, ***a*** ) . (12.10)

An often used alternative is to use a deep neural network to learn these
features instead. Doing so is also known as *deep reinforcement learning* . [6] 6 Note that often non-Bayesian neural
networks (i.e., point estimates of the

We will now apply the derivation from the previous section directly to weights) are used. In the final chapter,

chapter 13, we will explore the benefits

Q-learning. For Q-learning, after observing the transition ( ***x***, ***a***, *r*, ***x*** *[′]* ),

of using Bayesian neural networks.

the loss function is given as


*ℓ* ( ***θ*** ; ***x***, ***a***, *r*, ***x*** *[′]* ) = . 1

2


2
� *r* + *γ* max ***a*** *[′]* *∈A* *[Q]* *[⋆]* [(] ***[x]*** *[′]* [,] ***[ a]*** *[′]* [;] ***[ θ]*** [old] [)] *[ −]* *[Q]* *[⋆]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** [)] � . (12.11)


Here, we simply use Bellman’s optimality equation (10.33) to estimate
*q* *[⋆]* ( ***x***, ***a*** ), instead of the estimation of *v* *[π]* ( ***x*** ) using Bellman’s expectation
equation for TD-learning. The difference between the current approximation and the optimization target,

*δ* B = . *r* + *γ* max ***a*** *[′]* *∈A* *[Q]* *[⋆]* [(] ***[x]*** *[′]* [,] ***[ a]*** *[′]* [;] ***[ θ]*** [old] [)] *[ −]* *[Q]* *[⋆]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** [)] [,] (12.12)

is called *Bellman error* . Analogously to TD-learning, [7] we obtain the 7 compare to eq. (12.8)
gradient update,

***θ*** *←* ***θ*** *−* *α* *t* ***∇*** ***θ*** *ℓ* ( ***θ*** ; ***x***, ***a***, *r*, ***x*** *[′]* ) (12.13)


1
= ***θ*** *−* *α* *t* ***∇*** ***θ***
2


2
� *r* + *γ* max ***a*** *[′]* *∈A* *[Q]* *[⋆]* [(] ***[x]*** *[′]* [,] ***[ a]*** *[′]* [;] ***[ θ]*** [old] [)] *[ −]* *[Q]* *[⋆]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** [)] � using the definition of *ℓ* (12.11)


= ***θ*** + *α* *t* *δ* B ***∇*** ***θ*** *Q* *[⋆]* ( ***x***, ***a*** ; ***θ*** ) . (12.14) using the chain rule

When using a neural network to learn *Q* *[⋆]*, we can use automatic differentiation to obtain unbiased gradient estimates. In the case of linear
function approximation, we can compute the gradient exactly,

= ***θ*** + *α* *t* *δ* B ***∇*** ***θ*** ***θ*** *[⊤]* ***ϕ*** ( ***x***, ***a*** ) using the linear approximation of *Q* *[⋆]*

(12.10)
= ***θ*** + *α* *t* *δ* B ***ϕ*** ( ***x***, ***a*** ) . (12.15)

In the tabular setting, this algorithm is the same as Q-learning and, in
particular, converges to the true Q-function *q* *[⋆]* . [8] There are few such 8 see theorem 11.14
results in the approximate setting. Usage in practice indicates that
using an approximation of the value function “should be fine” when a
“rich-enough” class of functions is used.

model-free approximate reinforcement learning 219


(1,-1) (1,-1) (1,-1) (1,0)


(1,-1) (1,-1)


(-1,-1) 1 2 3 4 5 6 7 (1,-1)

(-1,-1) (-1,-1) (-1,0) (-1,-1) (-1,-1) (-1,-1)


Figure 12.1: MDP studied in exercise 12.2. Each arrow marks a (deterministic) transition and is labeled with
( action, reward ) .



*12.2.1* *Heuristics*

The vanilla stochastic semi-gradient descent is very slow. In this subsection, we will discuss some “tricks of the trait” to improve its perfor
mance.

There are mainly two problems. The first problem is that, as mentioned previously, the bootstrapping estimate changes after each iteration. As we are trying to learn an approximate value function that
depends on the bootstrapping estimate, this means that the optimization target is “moving” between iterations. In practice, moving targets
lead to stability issues. The first family of techniques we discuss here

220 probabilistic artificial intelligence

aim to “stabilize” the optimization targets.

One such technique is called *neural fitted Q-iteration* or *deep Q-networks*
(DQN). [9] DQN updates the neural network used for the approximate 9 Volodymyr Mnih, Koray Kavukcuoglu,

bootstrapping estimate infrequently to maintain a constant optimiza- David Silver, Andrei A Rusu, Joel Ve
ness, Marc G Bellemare, Alex Graves,

tion target across multiple episodes. How this is implemented exactly Martin Riedmiller, Andreas K Fidjeland,
varies. One approach is to clone the neural network and maintain one Georg Ostrovski, et al. Human-level con
trol through deep reinforcement learn
changing neural network (“online network”) for the most recent es- ing. *nature*, 518(7540):529–533, 2015
timate of the Q-function which is parameterized by ***θ***, and one fixed
neural network (“target network”) used as the target which is parameterized by ***θ*** [old] and which is updated infrequently.

This can be implemented by maintaining a data set *D* of observed
transitions (the so-called *replay buffer* ) and then “every once in a while”
(e.g., once *|D|* is large enough) solving a regression problem, where the
labels are determined by the target network. This yields a loss term
where the target is fixed across all transitions in the replay buffer *D*,

###### ℓ DQN ( θ ; D ) = . 1 ∑

2
( ***x***, ***a***, *r*, ***x*** *[′]* ) *∈D*


*ℓ* DQN ( ***θ*** ; *D* ) = . 1


2
� *r* + *γ* max ***a*** *[′]* *∈A* *[Q]* *[⋆]* [(] ***[x]*** *[′]* [,] ***[ a]*** *[′]* [;] ***[ θ]*** [old] [)] *[ −]* *[Q]* *[⋆]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** [)] � .

(12.16)


The loss can also be interpreted (in an online sense) as performing
regular Q-learning with the modification that the target network ***θ*** [old] is
not updated to ***θ*** after every observed transition, but instead only after
observing *|D|* -many transitions. This technique is known as *experience*
*replay* . Another approach is *Polyak averaging* where the target network
is gradually “nudged” by the neural network used to estimate the Q
function.

Now, observe that the estimates *Q* *[⋆]* are noisy estimates of *q* *[⋆]* and consider the term,

max ***a*** *[′]* *∈A* *[q]* *[⋆]* [(] ***[x]*** *[′]* [,] ***[ a]*** *[′]* [)] *[ ≈]* [max] ***a*** *[′]* *∈A* *[Q]* *[⋆]* [(] ***[x]*** *[′]* [,] ***[ a]*** *[′]* [;] ***[ θ]*** [old] [)] [,]

from the loss function (12.16). This term maximizes over a noisy estimate of *q* *[⋆]*, which leads to a biased estimate of max *q* *[⋆]* . The fact that
the update rules can be affected by inaccuracies (i.e., noise in the estimates) of the learned Q-function is known as the *“maximization bias”*
(see fig. 12.2).

*Double DQN* (DDQN) is an algorithm that addresses this maximization
bias. [10] Instead of picking the optimal action with respect to the old
network, it picks the optimal action with respect to the new network,

###### ℓ DDQN ( θ ; D ) = . 1 ∑

2
( ***x***, ***a***, *r*, ***x*** *[′]* ) *∈D*


*ℓ* DDQN ( ***θ*** ; *D* ) = . 1


2
*r* + *γQ* *[⋆]* ( ***x*** *[′]*, ***a*** *[⋆]* ( ***x*** *[′]* ; ***θ*** ) ; ***θ*** [old] ) *−* *Q* *[⋆]* ( ***x***, ***a*** ; ***θ*** )
� �

(12.17)

model-free approximate reinforcement learning 221

|ias as function of state|Col2|
|---|---|
|maxQ⋆(x, a) −maxq⋆(x, a) a a||
|||


*−* 1

1

0

*−* 1


0


2

0

*−* 2

2

0


True value and an estimate


Double-Q estimate

Double-Q estimate

|ax a Q⋆(x, a) −|−max a q|q⋆(x, a)|
|---|---|---|
||||
||||

|q⋆(x, a)|Col2|
|---|---|
|Q⋆(x, a)||
|||


4

2


4 max *a* *Q* *[⋆]* ( *x*, *a* ) *−*


2

0


max *a* *q* *[⋆]* ( *x*, *a* )

Double-Q estimate


0


|Col1|Col2|Q⋆(x, a)|
|---|---|---|
||q⋆(x, a)||


*−* 6 *−* 4 *−* 2 0 2 4 6

state


All estimates and max

2

0

*−* 2

2 max *a* *Q* *[⋆]* ( *x*, *a* )

0

4

2

0

*−* 6 *−* 4 *−* 2 0 2 4 6

state


where ***a*** *[⋆]* ( ***x*** *[′]* ; ***θ*** ) = . arg max *Q* *[⋆]* ( ***x*** *[′]*, ***a*** *[′]* ; ***θ*** ) . (12.18)
***a*** *[′]* *∈A*

Intuitively, this change ensures that the evaluation of the target network is consistent with the updated Q-function, which makes it more
difficult for the algorithm to be affected by noise.

Similarly to DQN this can also be interpreted as the online update,

***θ*** *←* ***θ*** + *α* *t* *r* + *γQ* *[⋆]* ( ***x*** *[′]*, ***a*** *[⋆]* ( ***x*** *[′]* ; ***θ*** ) ; ***θ*** [old] ) *−* *Q* *[⋆]* ( ***x***, ***a*** ; ***θ*** ) ***∇*** ***θ*** *Q* *[⋆]* ( ***x***, ***a*** ; ***θ*** )
� �

(12.19)

after observing a single transition ( ***x***, ***a***, *r*, ***x*** *[′]* ) where while differentiating, ***a*** *[⋆]* ( ***x*** *[′]* ; ***θ*** ) is treated as constant with respect to ***θ*** . ***θ*** [old] is then
updated to ***θ*** after observing *|D|* -many transitions.


*−* 6 *−* 4 *−* 2 0 2 4 6

state

Figure 12.2: Illustration of overestimation during learning. In each state (xaxis), there are 10 actions. The left column shows the true values *v* *[⋆]* ( *x* ) (purple line). All true action values are defined by *q* *[⋆]* ( *x*, *a* ) = . *v* *⋆* ( *x* ) . The green
line shows estimated values *Q* *[⋆]* ( *x*, *a* ) for
one action as a function of state, fitted to the true value at several sampled states (green dots). The middle column plots show all the estimated values (green), and the maximum of these
values (dashed black). The maximum is
higher than the true value (purple, left
plot) almost everywhere. The right column plots show the difference in red.
The blue line in the right plots is the estimate used by Double Q-learning with
a second set of samples for each state.
The blue line is much closer to zero, indicating less bias. The three rows correspond to different true functions (left,
purple) or capacities of the fitted function (left, green). Taken from “Deep
reinforcement learning with double qlearning” (Van Hasselt et al.).



222 probabilistic artificial intelligence

*12.3* *Policy Approximation*

Q-learning defines a policy implicitly by

***π*** ( ***x*** ) = . arg max *Q* *[⋆]* ( ***x***, ***a*** ) . (12.20)
***a*** *∈A*

Q-learning also maximizes over the set of all actions in its update step
while learning the Q-function. This is intractable for large and, in
particular, continuous action spaces. A natural idea to escape this limitation is to immediately learn an approximate parameterized policy,

***π*** ( ***x*** ) *≈* ***π*** ( ***x*** ; ***θ*** ) = . ***π*** ***θ*** ( ***x*** ) . (12.21)

Methods that find an approximate policy are also called *policy search*
*methods* or *policy gradient methods* . Policy gradient methods use randomized policies for encouraging exploration.



*12.3.1* *Estimating Policy Values*

We will begin by attributing a “value” to a policy. Recall the definition of the discounted payoff *G* *t* from time *t*, which we are aiming to

maximize,

∞
###### G t = ∑ γ [m] R t + m . see eq. (10.5)

*m* = 0

We define *G* *t* : *T* to be the *bounded discounted payoff* until time *T*,

*T* *−* *t*
###### G t : T = . ∑ γ [m] R t + m . (12.22)

*m* = 0

Based on these two random variables, we define the policy value func
tion.

**Definition 12.4** (Policy value function) **.** The *policy value function*,


�


*j* ( *π* ) = . **E** *π* [ *G* 0 ] = **E** *π*


∞
###### ∑ γ [t] R t
� *t* = 0


, (12.23)

model-free approximate reinforcement learning 223

measures the expected discounted payoff of policy *π* . We also define
the bounded variant,


�


*j* *T* ( *π* ) = . **E** *π* [ *G* 0: *T* ] = **E** *π*


*T*
###### ∑ γ [t] R t
� *t* = 0


. (12.24)


For simplicity, we will abbreviate *j* ( ***θ*** ) = . *j* ( *π* ***θ*** ) .

Naturally, we want to maximize *j* ( ***θ*** ) . That is, we want to find,

***θ*** *[⋆]* = . arg max *j* ( ***θ*** ) . (12.25)
***θ***

It turns out that this optimization problem is non-convex.

Let us see how *j* ( ***θ*** ) can be evaluated to understand the optimization
problem better. We will again use a Monte Carlo estimate. Recall that
a fixed ***θ*** induces a unique Markov chain, which can be simulated. In
the episodic setting, each episode (also called *rollout* ) of length *T* yields
an independently sampled trajectory,

. ( *i* )
*τ* [(] *[i]* [)] = (( ***x*** 0 [,] ***[ a]*** 0 [(] *[i]* [)] [,] *[ r]* 0 [(] *[i]* [)] [,] ***[ x]*** 1 [(] *[i]* [)] [)] [,] [ (] ***[x]*** 1 [(] *[i]* [)] [,] ***[ a]*** 1 [(] *[i]* [)] [,] *[ r]* 1 [(] *[i]* [)] [,] ***[ x]*** 2 [(] *[i]* [)] [)] [, . . .] [ )] (12.26)

Simulating *m* rollouts yields the samples,

*τ* [(] [1] [)], . . ., *τ* [(] *[m]* [)] *∼* Π ***θ***, (12.27)

where Π ***θ*** is the distribution of all trajectories (i.e., rollouts) of the
Markov chain induced by policy *π* ***θ*** . We denote the (bounded) discounted payoff of the *i* -th rollout by

*T*
###### g 0: [(] [i] [)] T = . ∑ γ [t] r t [(] [i] [)] (12.28)

*t* = 0

where *r* *t* [(] *[i]* [)] is the reward at time *t* of the *i* -th rollout. Using a Monte
Carlo approximation, we can then estimate *j* *T* ( ***θ*** ) . Moreover, due to
the exponential discounting of future rewards, it is reasonable to approximate the policy value function using bounded trajectories,


*j* ( ***θ*** ) *≈* *j* *T* ( ***θ*** ) *≈* [1]

*m*


*m*
###### ∑ g 0: [(] [i] [)] T [.] (12.29)

*i* = 1


*12.3.2* *Reinforce Gradient*

Let us now formally define the distribution over trajectories Π ***θ*** that
we used in the previous section. We can specify the probability of a
specific trajectory *τ* under a policy *π* ***θ*** by

*T*
###### Π θ ( τ ) = p ( x 0 ) ∏ π θ ( a t | x t ) p ( x t + 1 | x t, a t ) . (12.30)

*t* = 0

224 probabilistic artificial intelligence

For optimizing *j* ( ***θ*** ) we need to obtain unbiased estimates of its gradi
ent,

***∇*** ***θ*** *j* ( ***θ*** ) *≈* ***∇*** ***θ*** *j* *T* ( ***θ*** ) = ***∇*** ***θ*** **E** *τ* *∼* Π ***θ*** [ *G* 0 ] . (12.31)


Note that the expectation integrates over the measure Π ***θ***, which depends on the parameter ***θ*** . Thus, we cannot move the gradient operator
inside the expectation as we have done previously in remark 1.24. This
should remind you of the reparameterization trick (see eq. (5.73)) that
we used to solve a similar gradient in the context of variational inference. In this context, however, we cannot apply the reparameterization
trick. [11] Fortunately, there is another way of estimating this gradient. 11 This is because the distribution Π ***θ***
is generally not reparameterizable. We

**Theorem 12.5** (Score gradient estimator) **.** *Under some regularity assump-* will, however, see that reparameteri
zation gradients are also useful in re
*tions, we have*

inforcement learning. See, .e.g, sections 12.4.5 and 13.1.2.


**Theorem 12.5** (Score gradient estimator) **.** *Under some regularity assump-*
*tions, we have*


***∇*** ***θ*** **E** *τ* *∼* Π ***θ*** [ *G* 0 ] = **E** *τ* *∼* Π ***θ*** [ *G* 0 ***∇*** ***θ*** log Π ***θ*** ( *τ* )] . (12.32)


*This estimator of the gradient is called the* score gradient estimator *.*

*Proof.* To begin with, let us look at the so-called *score function*,

***∇*** ***θ*** log Π ***θ*** ( *τ* ) . (12.33)

Using the chain rule, yields

***∇*** ***θ*** log Π ***θ*** ( *τ* ) = ***[∇]*** ***[θ]*** [Π] ***[θ]*** [(] *[τ]* [)] . (12.34)

Π ***θ*** ( *τ* )

By rearranging the terms, we obtain

***∇*** ***θ*** Π ***θ*** ( *τ* ) = Π ***θ*** ( *τ* ) ***∇*** ***θ*** log Π ***θ*** ( *τ* ) . (12.35)

This is called the *score function trick* .

Now, assuming that state and action spaces are continuous, we obtain

***∇*** ***θ*** **E** *τ* *∼* Π ***θ*** [ *G* 0 ] = ***∇*** ***θ*** Π ***θ*** ( *τ* ) *·* *G* 0 *dτ* using the definition of expectation

� (1.23b)
= ***∇*** ***θ*** Π ***θ*** ( *τ* ) *·* *G* 0 *dτ* using the regularity assumptions to
�
swap gradient and integral
= *G* 0 *·* Π ***θ*** ( *τ* ) ***∇*** ***θ*** log Π ***θ*** ( *τ* ) *dτ* using the score function trick (12.35)
�

= **E** *τ* *∼* Π ***θ*** [ *G* 0 ***∇*** ***θ*** log Π ***θ*** ( *τ* )] . interpreting the integral as an
expectation over Π ***θ***

Intuitively, maximizing *j* ( ***θ*** ) increases the probability of policies with
high returns and decreases the probability of policies with low returns.

To use the score gradient estimator for estimating the gradient, we
need to compute ***∇*** ***θ*** log Π ***θ*** ( *τ* ) .

***∇*** ***θ*** log Π ***θ*** ( *τ* )

model-free approximate reinforcement learning 225


�


using the definition of the distribution
over trajectories Π ***θ***


= ***∇*** ***θ***


�


*T* *T*
###### log p ( x 0 ) + ∑ log π θ ( a t | x t ) + ∑ log p ( x t + 1 | x t, a t )

*t* = 0 *t* = 0


*T* *T*
###### = ∇ θ log p ( x 0 ) + ∑ ∇ θ log π θ ( a t | x t ) + ∑ ∇ θ log p ( x t + 1 | x t, a t )

*t* = 0 *t* = 0

*T*
###### = ∑ ∇ θ log π θ ( a t | x t ) . (12.36) using that the first and third term are

*t* = 0 independent of ***θ***

When using a neural network for the parameterization of the policy *π*,
we can use automatic differentiation to obtain unbiased gradient estimates. However, it turns out that the variance of these estimates is very
large. Using so-called *baselines* can reduce the variance dramatically.

**Lemma 12.6** (Score gradients with baselines) **.** *We have,*

**E** *τ* *∼* Π ***θ*** [ *G* 0 ***∇*** ***θ*** log Π ***θ*** ( *τ* )] = **E** *τ* *∼* Π ***θ*** [( *G* 0 *−* *b* ) ***∇*** ***θ*** log Π ***θ*** ( *τ* )] . (12.37)

*Here, b* *∈* **R** *is called a* baseline *.*

*Proof.* For the term to the right, we have due to linearity of expectation
(1.24),

**E** *τ* *∼* Π ***θ*** [( *G* 0 *−* *b* ) ***∇*** ***θ*** log Π ***θ*** ( *τ* )] = **E** *τ* *∼* Π ***θ*** [ *G* 0 ***∇*** ***θ*** log Π ***θ*** ( *τ* )]

*−* **E** *τ* *∼* Π ***θ*** [ *b* *·* ***∇*** ***θ*** log Π ***θ*** ( *τ* )] .

Thus, it remains to show that the second term is zero,


**E** *τ* *∼* Π ***θ*** [ *b* *·* ***∇*** ***θ*** log Π ***θ*** ( *τ* )] = *b* *·* Π ***θ*** ( *τ* ) ***∇*** ***θ*** log Π ***θ*** ( *τ* ) *dτ* using the definition of expectation
� (1.23b)



[Π] ***[θ]*** [(] *[τ]* [)]
= *b* *·* Π ***θ*** ( *τ* ) ***[∇]*** ***[θ]***
� Π ( *τ* )


*dτ* substituting the score function (12.33),
Π ***θ*** ( *τ* ) “undoing the score function trick”


= *b* *·* ***∇*** ***θ*** Π ***θ*** ( *τ* ) *dτ* Π ***θ*** ( *τ* ) cancels
�


= *b* *·* ***∇*** ***θ***


Π ***θ*** ( *τ* ) *dτ*
�


= *b* *·* ***∇*** ***θ*** 1 = 0. integrating a PDF over its domain is 1
and the derivative of a constant is 0


226 probabilistic artificial intelligence









model-free approximate reinforcement learning 227



(12.32)





. using a state-dependent baseline (12.40)




Performing stochastic gradient descent with the score gradient estimator and downstream returns is known as the *REINFORCE algorithm*
shown in alg. 12.11. [12] 12 Ronald J Williams. Simple statistical
gradient-following algorithms for connectionist reinforcement learning. *Ma-*
*chine learning*, 8(3):229–256, 1992
**Al** **g** **orithm 12.11:** REINFORCE al g orithm

**1** initialize policy weights ***θ***

**2** `repeat`

**3** generate an episode (i.e., rollout) to obtain trajectory *τ*

**4** `for` *t* = 0 `to` *T* `do`

**5** set *G* *t* : *T* to the downstream return from time *t*

**6** ***θ*** *←* ***θ*** + *ηγ* *[t]* *G* *t* : *T* ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *t* *|* ***x*** *t* ) `//` (12.44)

**7** `until` `converged`

228 probabilistic artificial intelligence



The variance of REINFORCE can be reduced further. A common tech
nique is to subtract a term *b* *t* from the downstream returns,


�


***∇*** ***θ*** *j* ( ***θ*** ) = **E** *τ* *∼* Π ***θ***


*T*
###### ∑ γ [t] ( G t : T − b t ) ∇ θ log π θ ( a t | x t )
� *t* = 0


. (12.46)


For example, we can subtract the mean reward to go,


*b* *t* = . 1

*T*


*T* *−* 1
###### ∑ G t : T . (12.47)

*t* = 0


The main advantage of policy gradient methods such as REINFORCE
is that they can be used in continuous action spaces. However, REINFORCE is not guaranteed to find an optimal policy. Even when
operating in very small domains, REINFORCE can get stuck in local
optima.

Typically, policy gradient methods are slow due to the large variance in
the score gradient estimates. Because of this, they need to take small
steps and require many rollouts of a Markov chain. Moreover, we
cannot reuse data from previous rollouts, as policy gradient methods
are fundamentally on-policy. [13] 13 This is because the score gradient estimator is used to obtain gradients of the
Next, we will combine value approximation techniques like Q-learning policy value function with respect to the
*current* policy.
and policy gradient methods, leading to actor-critic methods, which
are widely used.


model-free approximate reinforcement learning 229

*12.4* *Actor-Critic Methods*

Actor-Critic methods reduce the variance of policy gradient estimates
by using ideas from value function approximation. They use function
approximation both to approximate value functions and to approximate policies. The goal is for these algorithms to scale to reinforcement learning problems, where we both have large state spaces and
potentially large action spaces.

*12.4.1* *Advantage Function*

A key concept of actor-critic methods is the advantage function.

**Definition 12.13** (Advantage function) **.** Given a policy *π*, the *advantage*
*function*,

*a* *[π]* ( ***x***, ***a*** ) = . *q* *π* ( ***x***, ***a*** ) *−* *v* *π* ( ***x*** ) (12.48)

= *q* *[π]* ( ***x***, ***a*** ) *−* **E** ***a*** *′* *∼* *π* ( ***x*** ) � *q* *[π]* ( ***x***, ***a*** *[′]* ) �, (12.49) using eq. (10.14)

measures the advantage of picking action ***a*** *∈A* when in state ***x*** *∈X*
over simply following policy *π* .

It follows immediately from eq. (12.49) that for any policy *π* and state
***x*** *∈X*, there exists an action ***a*** *∈A* such that *a* *[π]* ( ***x***, ***a*** ) is non-negative,

max ***a*** *∈A* *[a]* *[π]* [(] ***[x]*** [,] ***[ a]*** [)] *[ ≥]* [0.] (12.50)

Moreover, it follows directly from Bellman’s theorem (10.32) that

*π* is optimal *⇐⇒∀* ***x*** *∈X*, ***a*** *∈A* : *a* *[π]* ( ***x***, ***a*** ) *≤* 0. (12.51)

In other words, quite intuitively, *π* is optimal if and only if there is no
action that has an advantage in any state over the action that is played
by *π* .

Finally, we can re-define the greedy policy *π* *q* with respect to the stateaction value function *q* as

*π* *q* ( ***x*** ) = . arg max *a* ( ***x***, ***a*** ) . (12.52)
***a*** *∈A*

Observe that we have,

arg max *a* ( ***x***, ***a*** ) = arg max *q* ( ***x***, ***a*** ) *−* *v* ( ***x*** ) = arg max *q* ( ***x***, ***a*** ),
***a*** *∈A* ***a*** *∈A* ***a*** *∈A*

as *v* ( ***x*** ) is independent of ***a*** . This coincides with our initial definition
of greedy policies in eq. (10.29). Intuitively, the advantage function is
a shifted version of the state-action function *q* that is relative to 0. It
turns out that using this quantity instead, has numerical advantages.

230 probabilistic artificial intelligence

*12.4.2* *Policy Gradient Theorem*

Recall the score gradient estimator (12.43) that we had introduced in
the previous section,


.

�


***∇*** ***θ*** *j* *T* ( ***θ*** ) = **E** *τ* *∼* Π ***θ***


*T*
###### ∑ γ [t] G t : T ∇ θ log π θ ( a t | x t )
� *t* = 0


Previously, we have approximated the policy value function *j* ( ***θ*** ) by the
bounded policy value function *j* *T* ( ***θ*** ) . We said that this approximation
was “reasonable” due to the diminishing returns. Essentially, we have
“cut off the tails” of the policy value function. Let us now reinterpret
score gradients while taking into account the tails of *j* ( ***θ*** ) .

***∇*** ***θ*** *j* ( ***θ*** ) = lim (12.53)
*T* *→* ∞ ***[∇]*** ***[θ]*** *[ j]* *[T]* [(] ***[θ]*** [)]

∞
###### = ∑ E τ ∼ Π θ � γ [t] G t ∇ θ log π θ ( a t | x t ) �. substituting the score gradient estimator

*t* = 0 with downstream returns (12.43) and
using linearity of expectation (1.24)
Observe that because the expectations only consider downstream returns, we can disregard all data from the trajectory prior to time *t* . Let
us define

*τ* *t* :∞ = (( . ***x*** *t*, ***a*** *t*, *r* *t*, ***x*** *t* + 1 ), ( ***x*** *t* + 1, ***a*** *t* + 1, *r* *t* + 1, ***x*** *t* + 2 ), . . . ), (12.54)

as the trajectory from time step *t* . Then,

∞
###### = ∑ E τ t :∞ ∼ Π θ � γ [t] G t ∇ θ log π θ ( a t | x t ) �.

*t* = 0

We now condition on ***x*** *t* and ***a*** *t*,

∞
###### = ∑ E x t, a t � γ [t] E r t, τ t + 1:∞ [ G t | x t, a t ] ∇ θ log π θ ( a t | x t ) �. using that π θ ( a t | x t ) is a constant given

*t* = 0 *x* *t* and *a* *t*

Observe that averaging over the trajectories **E** *τ* *∼* Π ***θ*** [ *·* ] that are sampled
according to policy *π* ***θ*** is equivalent to our shorthand notation **E** *π* ***θ*** [ *·* ]
from eq. (10.6),

∞
###### = ∑ E x t, a t � γ [t] E π θ [ G t | x t, a t ] ∇ θ log π θ ( a t | x t ) �

*t* = 0

∞
###### = ∑ E x t, a t � γ [t] q [π] [θ] ( x t, a t ) ∇ θ log π θ ( a t | x t ) �. (12.55) using the definition of the Q-function

*t* = 0 (10.8)

It turns out that **E** ***x*** *t*, ***a*** *t* [ *q* *[π]* ***[θ]*** ( ***x*** *t*, ***a*** *t* )] exhibits much less variance than our
previous estimator **E** ***x*** *t*, ***a*** *t* [ *G* *t* *|* ***x*** *t*, ***a*** *t* ] .

**Theorem 12.14** (Policy gradient theorem) **.** *Policy gradients can be repre-*
*sented in terms of the Q-function,*

***∇*** ***θ*** *j* ( ***θ*** ) = *ρ* ***θ*** ( ***x*** ) *·* **E** ***a*** *∼* *π* ***θ*** ( *·|* ***x*** ) [ *q* *[π]* ***[θ]*** ( ***x***, ***a*** ) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *|* ***x*** )] *d* ***x*** (12.56)
�

*where*


model-free approximate reinforcement learning 231

= . **E** ***x*** *∼* *ρ* ***θ*** **E** ***a*** *∼* *π* ***θ*** ( *·|* ***x*** ) [ *q* *π* ***θ*** ( ***x***, ***a*** ) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *|* ***x*** )] (12.57) this is not “really” an expectation as *ρ* ***θ***
is not a distribution

∞
###### ρ θ ( x ) = . ∑ γ [t] P π θ ( X t = x ) (12.58)

*t* = 0


*is called the* discounted state occupancy measure *.* [14] 14 Depending on the reward setting,
there exist various variations of the policy gradient theorem. We derived the

*Proof.* The right term of eq. (12.55) can be expressed as variant for infinite-horizon discounted

payoffs. “Reinforcement learning: An

∞ introduction” (Sutton and Barto) derive
###### ∑ � P ( X t = x ) · E a t ∼ π θ ( ·| x ) � γ [t] q [π] [θ] ( x t, a t ) ∇ θ log π θ ( a t | x t ) � d x . the variant for undiscounted average re-wards.


*Proof.* The right term of eq. (12.55) can be expressed as


∞
###### ∑

*t* = 0


**P** ( **X** *t* = ***x*** ) *·* **E** ***a*** *t* *∼* *π* ***θ*** ( *·|* ***x*** ) � *γ* *[t]* *q* *[π]* ***[θ]*** ( ***x*** *t*, ***a*** *t* ) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *t* *|* ***x*** *t* ) � *d* ***x*** .
�


Equation (12.56) then follows by swapping the order of sum and integral and reorganizing terms.

Intuitively, *ρ* ***θ*** ( ***x*** ) measures how often we visit state ***x*** when following
policy *π* ***θ*** . It can be thought of as a “discounted frequency”. Importantly, *ρ* ***θ*** is not a probability distribution, as it is not normalized to
integrate to 1. Instead, *ρ* ***θ*** is what is often called a *finite measure* . Therefore, eq. (12.57) is not a real expectation!

This can be understood as averaging over all state-action pairs according to how often they occur under the policy *π* ***θ*** . The policy gradient
is therefore often expressed as

***∇*** ***θ*** *j* ( ***θ*** ) = . **E** ( ***x***, ***a*** ) *∼* *π* ***θ*** [ *q* *π* ***θ*** ( ***x***, ***a*** ) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *|* ***x*** )] (12.59)

where we use ( ***x***, ***a*** ) *∼* *π* ***θ*** as shorthand notation for the random process ***x*** *∼* *ρ* ***θ***, ***a*** *∼* *π* ***θ*** ( *· |* ***x*** ) . Again, eq. (12.59) is not a real expectation
but commonly expressed in this form.

Matching our intuition, according to the policy gradient theorem, maximizing *j* ( ***θ*** ) corresponds to increasing the probability of actions with
a large value and decreasing the probability of actions with a small
value, taking into account how often the resulting policy visits certain

states.


232 probabilistic artificial intelligence







However, we cannot use the policy gradient to calculate the gradient
exactly, as we do not know *q* *[π]* ***[θ]*** . Instead, we will use bootstrapping
estimates *Q* *[π]* ***[θ]*** of *q* *[π]* ***[θ]*** .

*12.4.3* *On-policy Actor-Critics*

Actor-Critic methods consist of two components:

- a parameterized policy, *π* ( ***a*** *|* ***x*** ; ***θ*** *π* ) = . *π* ***θ***, which is called (12.64)
*actor* ; and

- a value function approximation, *q* *[π]* ***[θ]*** ( ***x***, ***a*** ) *≈* *Q* *[π]* ***[θ]*** ( ***x***, ***a*** ; ***θ*** *Q* ), which is
called *critic* . In the following, we will abbreviate *Q* *[π]* ***[θ]*** by *Q* . (12.65)

In deep reinforcement learning, neural networks are used to parameterize both actor and critic. In principle, the actor-critic framework
allows scaling to both large state spaces and large action spaces. We




**approximation**

Figure 12.3: Illustration of one iteration
of actor-critic methods. The dependencies between the actors and critics are

shown as arrows. Methods differ in the

exact order in which actor and critic are

updated.

model-free approximate reinforcement learning 233

begin by discussing on-policy actor-critics.

One approach in the online setting (i.e., non-episodic setting), is to
simply use TD-learning for learning the critic. To learn the actor, we
use stochastic gradient descent with gradients obtained using single
samples from

**E** ( ***x***, ***a*** ) *∼* *π* ***θ*** � *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *t* *|* ***x*** *t* ) � (12.66) see the policy gradient theorem (12.59)

where *Q* is a bootstrapping estimate of *q* *[π]* ***[θ]*** . This algorithm is known
as *online actor-critic* and shown in alg. 12.16.

**Al** **g** **orithm 12.16:** Online actor-critic, OAC

**1** `for` *t* = 0 `to` ∞ `do`

**2** use *π* ***θ*** to obtain transition ( ***x***, ***a***, *r*, ***x*** *[′]* )

**3** ***θ*** *π* *←* ***θ*** *π* + *η* *t* *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) ***∇*** ***θ*** *π* log *π* ***θ*** ( ***a*** *|* ***x*** ) `//` (12.67)

**4** ***θ*** *Q* *←* ***θ*** *Q* + *η* *t* ( *r* + *γQ* ( ***x*** *[′]*, *π* ***θ*** ( ***x*** *[′]* ) ; ***θ*** *Q* ) *−* *Q* ( ***x***, ***a*** ; ***θ*** *Q* )) ***∇*** ***θ*** *Q* *Q* ( ***x***, ***a*** ; ***θ*** *Q* )

`//` (12.68)

Observe that eq. (12.68) corresponds to the TD-learning update rule
(12.8) when used with the Q-function. [15] Due to the use of TD-learning 15 The gradient with respect to ***θ*** *Q* apfor learning the critic, this algorithm is fundamentally on-policy. pears analogously to our derivation of
approximate Q-learning (12.15).

Crucially, by neglecting the dependence of the bootstrapping estimate
*Q* on the policy parameters ***θ*** *π*, we introduce bias in the gradient estimates. In other words, using the bootstrapping estimate *Q* means that
the resulting gradient direction might not be a valid ascent direction.
In particular, the actor is not guaranteed to improve. Still, it turns out
that under strong so-called “compatibility conditions” that are rarely
satisfied in practice, a valid ascent direction can be guaranteed.

To further reduce the variance of the gradient estimates, it turns out
that a similar approach to the baselines we discussed in the previous
section on policy gradient methods is useful. One common example,
is to subtract the state value function from estimates of the Q-function,

***θ*** *π* *←* ***θ*** *π* + *η* *t* ( *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) *−* *V* ( ***x*** ; ***θ*** *V* )) ***∇*** ***θ*** *π* log *π* ***θ*** ( ***a*** *|* ***x*** ) (12.69)

= ***θ*** *π* + *η* *t* *A* ( ***x***, ***a*** ; ***θ*** *A* ) ***∇*** ***θ*** *π* log *π* ***θ*** ( ***a*** *|* ***x*** ) (12.70) using the definition of the advantage
function (12.48)

where *A* ( ***x***, ***a*** ; ***θ*** *A* ) is a bootstrapped estimate of the advantage function
*a* *[π]* ***[θ]*** . This algorithm is known as *advantage actor-critic* (A2C). [16] Recall 16 Volodymyr Mnih, Adria Puig
that the Q-function is an absolute quantity, whereas the advantage domenech Badia, Mehdi Mirza, Alex

Graves, Timothy Lillicrap, Tim Harley,

function is a relative quantity, where the sign is informative for the David Silver, and Koray Kavukcuoglu.
gradient direction. Intuitively, an absolute value is harder to estimate Asynchronous methods for deep re
inforcement learning. In *International*
*conference* *on* *machine* *learning*, pages
1928–1937. PMLR, 2016

234 probabilistic artificial intelligence

than the sign. Actor-Critic methods are therefore often implemented
with respect to the advantage function rather than the Q-function.

Taking a step back, observe that the policy gradient methods such as
REINFORCE generally have *high variance* in their gradient estimates.
However, due to using Monte Carlo estimates of *G* *t*, the gradient estimates are *unbiased* . In contrast, using a bootstrapped Q-function to
obtain gradient estimates yields estimates with a *smaller variance*, but

those estimates are *biased* . We are therefore faced with a *bias-variance*

*tradeoff* . A natural approach is therefore to blend both gradient estimates to allow for effectively trading bias and variance. This leads to
algorithms such as *generalized advantage estimation* (GAE/GAAE). [17] 17 John Schulman, Philipp Moritz, Sergey
Levine, Michael Jordan, and Pieter
Abbeel. High-dimensional continuous

*Improving sample efficiency* Actor-Critic methods generally suffer from control using generalized advantage eslow sample efficiency. When also using an on-policy method, actor- timation. *arXiv preprint arXiv:1506.02438*,
critics typically need an extremely large number of interactions before 2015b
learning a near-optimal policy, because they cannot reuse past data.
Allowing to reuse past data is a major advantage of off-policy methods
like Q-learning.


One well-known variant that slightly improves the sample efficiency is
*trust-region policy optimization* (TRPO). [18] TRPO uses multiple iterations, 18 John Schulman, Sergey Levine, Pieter

where in each iteration a fixed critic is used to optimize the policy. [19] Abbeel, Michael Jordan, and Philipp

Moritz. Trust region policy optimiza
During iteration *k*, we have, tion. In *International conference on machine*

*learning*, pages 1889–1897. PMLR, 2015a


where in each iteration a fixed critic is used to optimize the policy. [19]


During iteration *k*, we have,


***θ*** *k* + 1 *←* arg max *J* ( ***θ*** *k*, ***θ*** ) subject to KL ( ***θ*** *∥* ***θ*** *k* ) *≤* *δ* (12.71)
***θ***


19 Intuitively, each iteration performs a
collection of gradient ascent steps.


where


*J* ( ***θ*** *k*, ***θ*** ) = . **E** ( ***x***, ***a*** ) *∼* *π* ***θ*** *k*


*π* ***θ*** ( ***a*** *|* ***x*** )
� *π* ***θ*** *k* ( ***a*** *|* ***x*** ) *[A]* *[π]* ***[θ]*** *[k]* [ (] ***[x]*** [,] ***[ a]*** [)] �. (12.72)


Notably, *J* is an expectation with respect to the *previous* policy *π* ***θ*** *k* and
the previous critic *A* *[π]* ***[θ]*** *[k]* . TRPO is an example of *importance weighting*
where the importance weights,

*π* ***θ*** ( ***a*** *|* ***x*** )
*π* ***θ*** *k* ( ***a*** *|* ***x*** ) [,]

are used to correct for using the previous policy. To be able to assume
that the fixed critic is a good approximation within a certain “trust
region” (i.e., one iteration), we impose the constraint,

KL ( ***θ*** *∥* ***θ*** *k* ) *≤* *δ*,

to optimize only in the neighborhood of the current policy. This constraint is also necessary for the importance weights not to blow up.

Intuitively, taking the expectation with respect to the previous policy *π* ***θ*** *k*, means that we can reuse data from rollouts within the same

model-free approximate reinforcement learning 235

iteration. TRPO allows reusing past data as long as it can still be
“trusted”. This makes TRPO “somewhat” off-policy. Fundamentally,
though, TRPO is still an on-policy method.

*Proximal policy optimization* (PPO) is a heuristic variant of TRPO that
often works well in practice. [20] 20 John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and
Oleg Klimov. Proximal policy op*12.4.4* *Off-policy Actor-Critics* timization algorithms. *arXiv preprint*
*arXiv:1707.06347*, 2017

In many applications, sample efficiency is crucial. Either because requiring too many interactions is computationally prohibitive or because obtaining sufficiently many samples for learning a near-optimal
policy is simply impossible. We therefore now want to look at a separate family of actor-critic methods, which are off-policy, and hence,
allow for the reuse of past data. These algorithms use the reparameterization gradient estimates, which we encountered before in the context
of variational inference, [21] instead of score gradient estimators. 21 see section 5.6.1

The off-policy methods that we discussed in the previous section can
be understood as performing a variant of *policy iteration*, where we use
an estimate of the value of the current policy and then try to improve
that policy based on this estimate. They mostly vary in how improving
the policy is traded with improving the estimate of its value. Fundamentally, these methods rely on policy evaluation. [22] 22 Policy evaluation is at the core of policy iteration. See alg. 10.17 for the

The techniques that we will introduce in this section are much more definition of policy iteration and sec
tion 10.2 for a summary of policy eval
closely related to value iteration, essentially making use of Bellman’s uation in the context of Markov decision
optimality principle to characterize the optimal policy. processes.

To begin with, let us assume that the policy ***π*** is deterministic. We
will later lift this restriction in section 12.4.5. Recall that our initial
motivation to consider policy gradient methods and then actor-critic
methods was the intractability of

###### ∑

( ***x***, ***a***, *r*, ***x*** *[′]* ) *∈D*


2
� *r* + *γ* max ***a*** *[′]* *∈A* *[Q]* [(] ***[x]*** *[′]* [,] ***[ a]*** *[′]* [;] ***[ θ]*** [old] [)] *[ −]* *[Q]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** [)] � see (12.16)


when the action space *A* was large. What if we simply replace the
exact maximum by a parameterized policy?

###### ∑

( ***x***, ***a***, *r*, ***x*** *[′]* ) *∈D*


2
� *r* + *γQ* ( ***x*** *[′]*, ***π*** ***θ*** *π* ( ***x*** *[′]* ) ; ***θ*** [old] *Q* [)] *[ −]* *[Q]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** *[Q]* [)] � . (12.73)


In particular, how should we update ***θ*** *π* ?

We want to train our parameterized policy to learn the maximization
over action, that is, [23] 23 Here, we already apply the improvement of DDQN to use the most-recent
***π*** ***θ*** *π* *≈* ***π*** *Q* = arg max *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) . (12.74) estimate of the Q-function for action se***a*** *∈A* lection (see eq. (12.17)).

236 probabilistic artificial intelligence

The key idea is that if we use a “rich enough” parameterization of
policies, selecting the greedy policy with respect to *Q* is equivalent to

***θ*** *[⋆]* *π* [=] [ arg max] **E** ***x*** *∼* *µ* � *Q* ( ***x***, ***π*** ( ***x*** ; ***θ*** *π* ) ; ***θ*** *Q* ) � (12.75)
***θ*** *π*

where *µ* ( ***x*** ) *>* 0 is an *exploration distribution* over states with full support. [24] We refer to this expectation by 24 We require full support to ensure that
all states are explored.

*J* *µ* ( ***θ*** *π* ; ***θ*** *Q* ) = . **E** ***x*** *∼* *µ* � *Q* ( ***x***, ***π*** ( ***x*** ; ***θ*** *π* ) ; ***θ*** *Q* ) �. (12.76)

Commonly, the exploration distribution *µ* is taken as the distribution
that samples states uniformly from a replay buffer. Note that we can
easily obtain unbiased gradient estimates of *J* *µ* with respect to ***θ*** *π* by
writing,

***∇*** ***θ*** *π* *J* *µ* ( ***θ*** *π* ; ***θ*** *Q* ) = **E** ***x*** *∼* *µ* � ***∇*** ***θ*** *π* *Q* ( ***x***, ***π*** ( ***x*** ; ***θ*** *π* ) ; ***θ*** *Q* ) �. (12.77) as we have seen in remark 1.24

Analogously to on-policy actor-critics (see eq. (12.66)), we use a bootstrapping estimate of *Q* . That is, we neglect the dependence of the
critic *Q* on the actor ***π*** ***θ***, and in particular, the policy parameters ***θ*** *π* .
We have seen that bootstrapping works with Q-learning, so there is
reason to hope that it will work in this context too. This allows us to
use the chain rule to compute the gradient,

***∇*** ***θ*** *π* *Q* ( ***x***, ***π*** ( ***x*** ; ***θ*** *π* ) ; ***θ*** *Q* ) = ***D*** ***θ*** *π* ***π*** ( ***x*** ; ***θ*** *π* ) *·* ***∇*** ***a*** *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) �� ***a*** = ***π*** ( ***x*** ; ***θ*** *π* ) [.]

(12.78)

This corresponds to evaluating the bootstrapping estimate of the Qfunction at ***π*** ( ***x*** ; ***θ*** *π* ) and obtaining a gradient estimate of the policy
estimate (e.g., through automatic differentiation). Note that as ***π*** ***θ*** is
vector-valued, ***D*** ***θ*** *π* ***π*** ( ***x*** ; ***θ*** *π* ) is the Jacobian of ***π*** ***θ*** evaluated at ***x*** .


Now that we have estimates of the gradient of our optimization target
*J* *µ*, it is natural to ask how we should select actions (based on ***π*** ***θ*** ) to
trade exploration and exploitation. In policy gradient techniques, we
rely on the randomness in the policy to explore, but now we consider
deterministic policies. As our method is off-policy, a simple idea in
continuous action spaces is to add Gaussian noise to the action selected by ***π*** ***θ*** — also known as *Gaussian noise “dithering”* . [25] This corre- 25 Intuitively, this “randomizes” the polsponds to an algorithm called *deep deterministic policy gradients* shown icy ***π*** ***θ*** .
in alg. 12.17. [26] 26 Timothy P Lillicrap, Jonathan J Hunt,
Alexander Pritzel, Nicolas Heess, Tom

This algorithm is essentially equivalent to Q-learning with function Erez, Yuval Tassa, David Silver, and

Daan Wierstra. Continuous control

approximation (e.g., DQN), [27] with the only exception that we replace with deep reinforcement learning. *arXiv*

the maximization over actions with the learned policy *π* ***θ*** . *preprint arXiv:1509.02971*, 2015


This algorithm is essentially equivalent to Q-learning with function
approximation (e.g., DQN), [27] with the only exception that we replace


the maximization over actions with the learned policy *π* ***θ*** . *preprint arXiv:*

27 see (12.16)


*Twin delayed DDPG* (TD3) is an extension of DDPG that uses two separate critic networks for predicting the maximum action and evaluating

model-free approximate reinforcement learning 237

**Al** **g** **orithm 12.17:** Dee p deterministic p olic y g radients, DDPG

**1** initialize ***θ*** *π*, ***θ*** *Q*, a (possibly non-empty) replay buffer *D* = ∅

**2** set ***θ*** [old] *π* = ***θ*** *π* and ***θ*** [old] *Q* [=] ***[ θ]*** *[Q]*

**3** `for` *t* = 0 `to` ∞ `do`

**4** observe state ***x***, pick action ***a*** = ***π*** ( ***x*** ; ***θ*** *π* ) + ***ϵ*** for ***ϵ*** *∼N* ( **0**, *λ* ***I***,)

**5** execute ***a***, observe *r* and ***x*** *[′]*

**6** add ( ***x***, ***a***, *r*, ***x*** *[′]* ) to the replay buffer *D*

**7** `if` `collected “enough” data` `then`
```
     // policy improvement step

```
**8** `for` `some iterations` `do`

**9** sample a mini-batch *B* of *D*

**10** for each transition in *B*, compute the label
*y* = *r* + *γQ* ( ***x*** *[′]*, ***π*** ( ***x*** *[′]* ; ***θ*** [old] *π* [)] [;] ***[ θ]*** [old] *Q* [)]
```
       // critic update

```
1
**11** ***θ*** *Q* *←* ***θ*** *Q* *−* *η* ***∇*** ***θ*** *Q* *B* [∑] [(] ***[x]*** [,] ***[a]*** [,] *[r]* [,] ***[x]*** *[′]* [,] *[y]* [)] *[∈]* *[B]* [(] *[Q]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** *[Q]* [)] *[ −]* *[y]* [)] [2]
```
       // actor update

```
1
**12** ***θ*** *π* *←* ***θ*** *π* + *η* ***∇*** ***θ*** *π* *B* [∑] [(] ***[x]*** [,] ***[a]*** [,] *[r]* [,] ***[x]*** *[′]* [,] *[y]* [)] *[∈]* *[B]* *[Q]* [(] ***[x]*** [,] ***[ π]*** [(] ***[x]*** [;] ***[ θ]*** *[π]* [)] [;] ***[ θ]*** *[Q]* [)]

**13** ***θ*** [old] *Q* *[←]* [(] [1] *[ −]* *[ρ]* [)] ***[θ]*** [old] *Q* [+] *[ ρ]* ***[θ]*** *[Q]*

**14** ***θ*** [old] *π* *←* ( 1 *−* *ρ* ) ***θ*** [old] *π* [+] *[ ρ]* ***[θ]*** *[π]*

238 probabilistic artificial intelligence

the policy. This addresses the maximization bias akin to Double-DQN.
TD3 also applies delayed updates to the actor network, which increases
stability.

*12.4.5* *Off-Policy Actor Critics with Randomized Policies*

We have seen that randomized policies naturally encourage exploration. With deterministic actor-critic methods like DDPG, we had

to inject Gaussian noise to enforce sufficient exploration. A natural
question is therefore whether we can also handle randomized policies
in this framework of off-policy actor-critics.

The key idea is to replace the squared loss of the critic,


1

2


2
� *r* + *γQ* ( ***x*** *[′]*, ***π*** ( ***x*** *[′]* ; ***θ*** [old] *π* [)] [;] ***[ θ]*** [old] *Q* [)] *[ −]* *[Q]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** *[Q]* [)] �, refer to the squared loss of Q-learning
(12.11)


which only considers the fixed action ***π*** ( ***x*** *[′]* ; ***θ*** [old] *π* [)] [ with an expected]
squared loss,

1 2
2 **[E]** ***[a]*** *[′]* *[∼]* *[π]* [(] ***[x]*** *[′]* [;] ***[θ]*** [old] *π* [)] � *r* + *γQ* ( ***x*** *[′]*, ***a*** *[′]* ; ***θ*** [old] *Q* [)] *[ −]* *[Q]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** *[Q]* [)] �, (12.79)

which considers a distribution over actions.

It turns out that we can still compute gradients of this expectation.

2
***∇*** ***θ*** *Q* **E** ***a*** *′* *∼* *π* ( ***x*** *′* ; ***θ*** old *π* [)] � *r* + *γQ* ( ***x*** *[′]*, ***a*** *[′]* ; ***θ*** [old] *Q* [)] *[ −]* *[Q]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** *[Q]* [)] �


= **E** ***a*** *′* *∼* *π* ( ***x*** *′* ; ***θ*** old *π* [)]


2 [�]
� ***∇*** ***θ*** *Q* � *r* + *γQ* ( ***x*** *[′]*, ***a*** *[′]* ; ***θ*** [old] *Q* [)] *[ −]* *[Q]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** *[Q]* [)] � . as we have seen in remark 1.24


Similarly to our definition of the Bellman error (12.12), we define by

*δ* B ( ***a*** *[′]* ) = . *r* + *γQ* ( ***x*** *′*, ***a*** *′* ; ***θ*** old *Q* [)] *[ −]* *[Q]* [(] ***[x]*** [,] ***[ a]*** [;] ***[ θ]*** *[Q]* [)] [,] (12.80)

the *Bellman error* for a fixed action ***a*** *[′]* . Using the chain rule, we obtain

= **E** ***a*** *′* *∼* *π* ( ***x*** *′* ; ***θ*** old *π* [)] �2 *δ* B ( ***a*** *[′]* ) *·* ***∇*** ***θ*** *Q* *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) �. (12.81)

This provides us with a method of obtaining unbiased gradient estimates. As we have done many times already, we can use automatic
differentiation to obtain gradient estimates of ***∇*** ***θ*** *Q* *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) .

We also need to reconsider the actor update. Recall that for deterministic policies,

***∇*** ***θ*** *π* *Q* ( ***x***, ***π*** ( ***x*** ; ***θ*** *π* ) ; ***θ*** *Q* ) = ***D*** ***θ*** *π* ***π*** ( ***x*** ; ***θ*** *π* ) *·* ***∇*** ***a*** *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) �� ***a*** = ***π*** ( ***x*** ; ***θ*** *π* ) [.] see (12.78)

When using a randomized policy, the second term turns into an expectation,

***∇*** ***θ*** *π* **E** ***a*** *∼* *π* ( ***x*** ; ***θ*** *π* ) � *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) �. (12.82)

model-free approximate reinforcement learning 239

Note that this expectation is with respect to a measure that depends
on the parameters ***θ*** *π*, which we are trying to optimize. We therefore cannot move the gradient operator inside the expectation as we
have done previously in remark 1.24. This is a problem that we have
already encountered several times. In the previous section on policy
gradients, we used the score gradient estimator. [28] Earlier, in chap- 28 see eq. (12.32)
ter 5 on variational inference we have already seen reparameterization
gradients. [29] Here, if our policy is reparameterizable, we can use the 29 see (5.74)
reparameterization trick from theorem 5.35!



As we have seen, not only Gaussians are reparameterizable. In general,
we called a distribution (i.e., in this context, a policy) reparameterizable iff ***a*** *∼* *π* ( ***x*** ; ***θ*** *π* ) is such that ***a*** = ***g*** ( ***ϵ*** ; ***x***, ***θ*** *π* ), where ***ϵ*** *∼* *ϕ* is an
independent random variable.

Then, we have,

***∇*** ***θ*** *π* **E** ***a*** *∼* *π* ( ***x*** ; ***θ*** *π* ) � *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) �

= **E** ***ϵ*** *∼* *ϕ* � ***∇*** ***θ*** *π* *Q* ( ***x***, ***g*** ( ***ϵ*** ; ***x***, ***θ*** *π* ) ; ***θ*** *Q* ) � (12.84) using the reparameterization trick (5.74)

= **E** ***ϵ*** *∼* *ϕ* � ***∇*** ***a*** *Q* ( ***x***, ***a*** ; ***θ*** *Q* ) �� ***a*** = ***g*** ( ***ϵ*** ; ***x***, ***θ*** *π* ) *[·]* ***[ ∇]*** ***[θ]*** *[π]* ***[ g]*** [(] ***[ϵ]*** [;] ***[ x]*** [,] ***[ θ]*** *[π]* [)] �. (12.85) using the chain rule

This allows us to obtain unbiased gradient estimates for reparameterizable policies. This technique does not only apply to continuous action
spaces. For discrete action spaces, there is the so-called *Gumbel-max*
*trick*, which we will not discuss in greater detail.

The algorithm that uses eq. (12.81) to obtain gradients for the critic
and reparameterization gradients for the actor is called *stochastic value*
*gradients* (SVG). [30] 30 Nicolas Heess, Greg Wayne, David Silver, Timothy Lillicrap, Yuval Tassa, and
Tom Erez. Learning continuous control
policies by stochastic value gradients.
*arXiv preprint arXiv:1510.09142*, 2015

240 probabilistic artificial intelligence

*12.4.6* *Off-policy Actor-Critics with Entropy Regularization*

In practice, algorithms like SVG often do not explore enough. A key
issue with relying on randomized policies for exploration is that they
might collapse to deterministic policies. That is, the algorithm might
quickly reach a local optimum, where all mass is placed on a single

action.

A simple trick that encourages a little bit of extra exploration is to
regularize the randomized policies “away” from putting all mass on
a single action. In other words, we want to encourage the policies to
exhibit some uncertainty. A natural measure of uncertainty is entropy,
which we have already seen several times. [31] This approach is known 31 see section 5.3
as *entropy regularization* and it leads to an analogue of Markov decision processes called *entropy-regularized Markov decision process*, where
suitably defined regularized state-action value functions (so-called *soft*
*value functions* ) are used. Canonically, entropy regularization is applied
to finite-horizon rewards (cf. remark 10.5), yielding the optimization
problem of maximizing

*j* *λ* ( ***θ*** ) = . *j* ( ***θ*** ) + *λ* H [ Π ***θ*** ] (12.86)

*T*
###### = ∑ E ( x t, a t ) ∼ π θ [ r ( x t, a t ) + λ H [ π θ ( · | x t )]], (12.87)

*t* = 0

where we have a preference for entropy in the actor distribution to
encourage exploration which is regulated by the temperature parameter *λ* . This is in contrast to the “standard” reinforcement learning
objective (here for finite-horizon rewards),

*T*
###### j ( θ ) = ∑ E ( x t, a t ) ∼ π θ [ r ( x t, a t )] (12.88)

*t* = 0

that we have been considering.


The same policy gradient and actor-critic methods such as DDPG and
SVG, which we have discussed, can be re-derived in this setting. The
resulting algorithm, *soft actor critic* (SAC), [32] is widely used. Due to its 32 Tuomas Haarnoja, Aurick Zhou, Pieter

off-policy nature, it is also relatively sample efficient. Abbeel, and Sergey Levine. Soft actor
critic: Off-policy maximum entropy deep
reinforcement learning with a stochastic


off-policy nature, it is also relatively sample efficient.


model-free approximate reinforcement learning 241



(12.90) distribution are “fixed”, that is, we

assume *p* ( ***x*** 0 *| O* 1: *T* ) = *p* ( ***x*** 0 ) and













mize forward-KL as we cannot sample
from Π *⋆* .








using eqs. (12.30) and (12.91) and
simplifying




242 probabilistic artificial intelligence





reward state will include sub-optimal ac


*12.4.7* *Summary*

In this chapter, we studied central ideas in actor-critic methods. We
have seen two main approaches to use policy-gradient methods. We
began, in section 12.3, by introducing the REINFORCE algorithm which
uses policy gradients and Monte Carlo estimation, but suffered from
large variance in the gradient estimates of the policy value function.

We have then seen a number of actor-critic methods such as A2C and

GAE behaving similarly to policy iteration that exhibit less variance,
but are very sample inefficient due to their on-policy nature. TRPO
improves the sample efficiency slightly, but not fundamentally.

We then discussed a second family of policy gradient techniques that
generalize Q-learning and are akin to value iteration. For reparameterizable policies, this led us to algorithms such as DDPG, TD3, SVG,

model-free approximate reinforcement learning 243


and SAC that are widely used and work quite well in practice. Importantly, they are significantly more sample efficient than on-policy
policy gradient methods, which often results in much faster learning
of a near-optimal policy.


Figure 12.4: Comparison of training
curves of a selection of on-policy and offpolicy policy gradient methods. Taken
from “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor” (Haarnoja
et al.).



244 probabilistic artificial intelligence


### *13* *Model-based Approximate Reinforcement Learning*

In this final chapter, we will revisit the model-based approach to reinforcement learning. We will see some advantages it offers over modelfree methods. In particular, we will use the machinery of Bayesian
learning, which we developed in the first chapters, to quantify uncertainty about our model and use this uncertainty for planning, exploration, and reliable decision-making.

To recap, in chapter 11, we began by discussing model-based reinforcement learning which attempts to learn the underlying Markov
decision process and then use it for planning. We have seen that in
the tabular setting, computing and storing the entire model is computationally expensive. This led us to consider the family of model-free
approaches, which learn the value function directly, and as such can
be considered more economical in the amount of data that they store.

In chapter 12, we saw that using function approximation, we were able
to scale model-free methods to very large state and action spaces. We
will now explore similar ideas in the model-based framework. Namely,
we will use function approximation to condense the representation
of our model of the environment. More concretely, we will learn an
approximate dynamics model *f* *≈* *p* and approximate rewards *r* .

There are a few ways in which the model-based approach is advantageous. First, if we have an accurate model of the environment, we can
safely use it for planning. However, in practice, we will rarely have
such a model. In fact, the accuracy of our model will depend on the
past experience of our agent and the region of the state-action space.
Understanding the uncertainty in our model of the environment is crucial for planning. In particular, quantifying uncertainty is necessary to
drive safe(!) exploration and avoid undesired states.

Moreover, modeling uncertainty in our model of the environment can

246 probabilistic artificial intelligence

be extremely useful in deciding where to explore. Learning a model
can therefore help to dramatically reduce the sample complexity over
model-free techniques. Often times this is crucial when developing
agents for real-world use as there, exploration is costly and potentially
dangerous.

Algorithm 13.1 describes the general approach to model-based reinforcement learning.

**Al** **g** **orithm 13.1:** Model-based reinforcement learnin g (outline)

**1** start with an initial policy *π* and no (or some) initial data *D*

**2** `for` `several episodes` `do`

**3** roll out policy *π* to collect data

**4** learn a model of the dynamics *f* and rewards *r* from data

**5** plan a new policy *π* based on the estimated models

We face three main challenges in model-based reinforcement learning.
First, given a fixed model, we need to perform planning to decide on
which actions to play. Second, we need to learn models *f* and *r* accurately and efficiently. Third, we need to effectively trade exploration
and exploitation. We will discuss these three topics in the following.

*13.1* *Planning*

There exists a large literature on planning in various settings. These
settings can be mainly characterized as

 - discrete or continuous action spaces;

 - fully- or partially observable state spaces;

 - constrained or unconstrained; and

 - linear or non-linear dynamics.

In chapter 10, we have already seen algorithms such as policy iteration
and value iteration, which can be used to solve planning exactly in
tabular settings. In the following, we will now focus on the setting of
continuous state and action spaces, fully observable state spaces, no
constraints, and non-linear dynamics.

*13.1.1* *Deterministic Dynamics*

To begin with, let us assume that our dynamics model is deterministic
and known. That is, given a state-action pair, we know the subsequent

state,

***x*** *t* + 1 = ***f*** ( ***x*** *t*, ***a*** *t* ) . (13.1)

model-based approximate reinforcement learning 247

We continue to focus on the setting of infinite-horizon discounted returns (10.5), which we have been considering throughout our discussion of reinforcement learning. This yields the objective,


max
***a*** 0:∞


∞
###### ∑ γ [t] r ( x t, a t ) such that x t + 1 = f ( x t, a t ) . (13.2)

*t* = 0


Now, because we are optimizing over an infinite time horizon, we cannot solve this optimization problem directly. This is the problem that
is ubiquitously studied in the area of *optimal control* . We will discuss
one central idea from optimal control that is widely used in modelbased reinforcement learning, and will later return to using this idea
for learning parametric policies in section 13.1.3.

The key idea of a classic algorithm from optimal control called *receding*
*horizon control* (RHC) or *model predictive control* (MPC) is to iteratively
plan over finite horizons. That is, in each round, we plan over a finite
time horizon *H* and carry out the first action.

**Al** **g** **orithm 13.2:** Model p redictive control, MPC

**1** `for` *t* = 0 `to` ∞ `do`

**2** observe ***x*** *t*

**3** plan over a finite horizon *H*,


***x*** *[⋆]*



Figure 13.1: Illustration of model predictive control in a deterministic transition
model. The agent starts in position ***x*** 0
and wants to reach ***x*** *[⋆]* despite the black
obstacle. We use the reward function
*r* ( ***x*** ) = *−∥* ***x*** *−* ***x*** *[⋆]* *∥* . The gray concentric circles represent the length of a single step. We plan with a time horizon of
*H* = 2. Initially, the agent does not “see”
the black obstacle, and therefore moves
straight towards the goal. As soon as the
agent sees the obstacle, the optimal trajectory is “replanned”. The dotted red
line corresponds to the optimal trajectory, the agent’s steps are shown in blue.


max
***a*** *t* : *t* + *H* *−* 1


*t* + *H* *−* 1
###### ∑ γ [τ] [−] [t] r ( x τ, a τ ) such that x τ + 1 = f ( x τ, a τ ) (13.3)

*τ* = *t*


**4** carry out action ***a*** *t*

Observe that the state ***x*** *τ* can be interpreted as a deterministic function
***x*** *τ* ( ***a*** *t* : *τ* *−* 1 ), which depends on all actions from time *t* to time *τ* *−* 1 and
the state ***x*** *t* . To solve the optimization problem of a single iteration, we
therefore need to maximize,

*t* + *H* *−* 1
###### J H ( a t : t + H − 1 ) = . ∑ γ [τ] [−] [t] r ( x τ ( a t : τ − 1 ), a τ ) . (13.4)

*τ* = *t*


It turns out that this optimization is non-convex. If the actions are
continuous and the dynamics and reward models are differentiable,
we can nevertheless obtain analytic gradients of *J* *H* . This can be done
using the chain rule and “backpropagating” through time, analogously
to backpropagation in neural networks. [1] Especially for large horizons 1 see section 7.1.4
*H*, this optimization problem becomes difficult to solve exactly due to
local optima and vanishing/exploding gradients.

Often, heuristic global optimization methods are used to optimize *J* *H* .
An example are *random shooting methods*, which find the optimal choice
among a set of random proposals.

248 probabilistic artificial intelligence

**Al** **g** **orithm 13.3:** Random shootin g methods

**1** generate *m* sets of random samples, ***a*** [(] *t* : *[i]* *t* [)] + *H* *−* 1

**2** pick the sequence of actions ***a*** [(] *t* : *[i]* *t* *[⋆]* + [)] *H* *−* 1 [where]

*i* *[⋆]* = arg max *J* *H* ( ***a*** [(] *t* : *[i]* *t* [)] + *H* *−* 1 [)] (13.5)
*i* *∈* [ *m* ]

*Monte Carlo tree search*, which is a planning method used for example
by “AlphaZero”, can be viewed as an advanced variant of a shooting

method.

Of course, obtaining a “good” set of randomly proposed action sequences is crucial. A naïve way of generating the proposals is to pick
them uniformly at random. This strategy, however, usually does not
perform very well as it corresponds to suggesting random walks of the
state space. Alternatives are to sample from a Gaussian distribution
or using the *cross-entropy method* which gradually adapts the sampling
distribution by reweighing samples according to the rewards they pro
duce.

A common problem of finite-horizon methods is that in the setting of
sparse rewards, there is often no signal that can be followed. You can
think of an agent that operates in a kitchen and tries to find a box
of candy. Yet, to get to this box, it needs to perform a large number
of actions. In particular, if this number of actions is larger than the
horizon *H*, then the local optimization problem of MPC will not take
into account the reward for choosing this long sequence of actions and
finding the box of candy. Thus, the box of candy will likely never be
found by our agent.

A solution to this problem is to amend a long-term value estimate to
the finite-horizon sum. The idea is to not only consider the rewards
attained *while* following the actions ***a*** *t* : *t* *H* *−* 1, but to also consider the
value of the final state ***x*** *t* + *H*, which estimates the discounted sum of
*future* rewards.

*t* + *H* *−* 1
###### J H ( a t : t + H − 1 ) = . ∑ γ [τ] [−] [t] r ( x τ ( a t : τ − 1 ), a τ ) + γ [H] V ( x t + H ) . (13.6)

*τ* = *t*

Intuitively, *γ* *[H]* *V* ( ***x*** *t* + *H* ) is estimating the tail of the infinite sum.

Observe that for *H* = 1, when we use the value function estimate

associated with this MPC controller, maximizing *J* *H* coincides with
using the greedy policy *π* *V* . That is,

***a*** *t* = arg max *J* 1 ( ***a*** ) = *π* *V* ( ***x*** *t* ) . (13.7)
***a*** *∈A*


Figure 13.2: Illustration of finite-horizon
planning with sparse rewards. When the
finite time horizon does not suffice to
“reach” a reward, the agent has no signal to follow.

model-based approximate reinforcement learning 249

Thus, by looking ahead for a single time step, we recover the approaches from the model-free setting in this model-based setting! Essentially, if we do not plan long-term and only consider the value estimate, the model-based setting reduces to the model-free setting. However, in the model-based setting, we are now able to use our model
of the transition dynamics to be more economical by considering the
downstream effects of picking a particular action now. [2] 2 This is one of the fundamental reasons for why model-based approaches
To obtain the value estimates, we can use the approaches we discussed are generally much more sample efficient than model-free methods.
in detail in section 11.3, such as TD-learning to obtain on-policy value
estimates and Q-learning to obtain off-policy value estimates. For large
state-action spaces, we can use their approximate variants, which we

discussed in section 12.1 and section 12.2.

*13.1.2* *Stochastic Dynamics*

How can we extend this approach to planning to a stochastic transition
model? A natural extension of model predictive control is to do what
is called *stochastic average approximation* (SAA) or *trajectory sampling* .
Like in MPC, we will still optimize over a deterministic sequence of
actions, but now we will average over all resulting trajectories.

**Al** **g** **orithm 13.4:** Tra j ector y sam p lin g

**1** `for` *t* = 0 `to` ∞ `do`

**2** observe ***x*** *t*

**3** optimize expected performance over a finite horizon *H*,


�

(13.8)


***a*** *t* : *t* + *H* *−* 1, *f*
�����


Figure 13.3: Illustration of trajectory
sampling. High-reward states are shown
in brighter colors. The agent iteratively
plans a finite number of time steps into
the future and picks the best initial action.


max
***a*** *t* : *t* + *H* *−* 1 **[E]** ***[x]*** *[t]* [+] [1:] *[t]* [+] *[H]*

**4** carry out action ***a*** *t*


*t* + *H* *−* 1
###### ∑ γ [τ] [−] [t] r τ + γ [H] V ( x t + H )
� *τ* = *t*


Intuitively, trajectory sampling optimizes over a much simpler object,
namely a deterministic sequence of actions of length *H*, than finding
a policy, which corresponds to finding an optimal decision tree mapping states to actions. Of course, using trajectory sampling (from an
arbitrary starting state) implies such a policy. However, in each step,
trajectory sampling only plans for a finite number of time steps.

Computing the expectation exactly involves solving a highdimensional integral of non-linear functions. Instead, a common approach is to use Monte Carlo estimates of this expectation. This approach is known as *Monte Carlo trajectory sampling* . The key issue with
using sampling based estimation is that the sampled trajectory (i.e.,

250 probabilistic artificial intelligence

sampled sequence of states) we obtain, depends on the actions we
pick. In other words, the measure we average over depends on the
decision variables — the actions. This is a problem that we have seen
several times already! It naturally suggests using the reparameterization trick. [3] 3 see theorem 5.35 and eq. (12.84)

Previously, we have used the reparameterization trick to reparameterize variational distributions (see section 5.6.1) and to reparameterize
policies (see section 12.4.5). It turns out that we can use the exact same
approach for reparameterizing the transition model. We say that a
(stochastic) transition model *f* is *reparameterizable* iff ***x*** *t* + 1 *∼* *f* ( ***x*** *t*, ***a*** *t* ) is
such that ***x*** *t* + 1 = ***g*** ( ***ϵ*** ; ***x*** *t*, ***a*** *t* ), where ***ϵ*** *∼* *ϕ* is a random variable that is
independent of ***x*** *t* and ***a*** *t* . We have already seen in example 12.18 (in
the context of stochastic policies) how a Gaussian transition model can
be reparameterized.

In this case, ***x*** *τ* is determined recursively by ***a*** *t* : *τ* *−* 1 and ***ϵ*** *t* : *τ* *−* 1,

***x*** *τ* = ***x*** *τ* ( ***ϵ*** *t* : *τ* *−* 1 ; ***a*** *t* : *τ* *−* 1 )
= . ***g*** ( ***ϵ*** *τ* *−* 1 ; ***g*** ( . . . ; ( ***ϵ*** *t* + 1 ; ***g*** ( ***ϵ*** *t* ; ***x*** *t*, ***a*** *t* ), ***a*** *t* + 1 ), . . . ), ***a*** *τ* *−* 1 ) . (13.9)

This allows us to obtain unbiased estimates of *J* *H* using Monte Carlo
approximation,


� *γ* *[τ]* *[−]* *[t]* *r* ( ***x*** *τ* ( ***ϵ*** *t* [(] : *[i]* *τ* [)] *−* 1 [;] ***[ a]*** *[t]* [:] *[τ]* *[−]* [1] [)] [,] ***[ a]*** *[τ]* [)]

(13.10)
+ *γ* *[H]* *V* ( ***x*** *t* + *H* )
�


*t* + *H* *−* 1
###### ∑

*τ* = *t*


*J* *H* ( ***a*** *t* : *t* + *H* *−* 1 ) *≈* [1]

*m*


*m*
###### ∑

*i* = 1


iid
where ***ϵ*** *t* [(] : *[i]* *t* [)] + *H* *−* 1 *∼* *ϕ* are independent samples. To optimize this approximation we can again compute analytic gradients or use shooting
methods as we have discussed in section 13.1.1 for deterministic dy
namics.

*13.1.3* *Parametric Policies*

When using algorithms such as model predictive control for planning,
planning needs to be done online before each time we take an action.
This can be expensive. Especially when the time horizon is large, or we
encounter similar states many times (leading to “repeated optimization problems”), it can be beneficial to “store” the planning decision in
a (deterministic) policy,

***a*** *t* = ***π*** ( ***x*** *t* ; ***θ*** ) = . ***π*** ***θ*** ( ***x*** *t* ) . (13.11)

This policy can then be trained offline and evaluated cheaply online.

This is akin to something that we have discussed in detail in the previous chapter. Observe that when extending Q-learning to large action

model-based approximate reinforcement learning 251

spaces, we faced a similar problem. This then led us to discuss policy
gradient and actor-critic methods. Recall that in Q-learning, we want
to follow the greedy policy,

***π*** *Q* = arg max *Q* ( ***x***, ***a*** ; ***θ*** *Q* ), see eq. (12.20)
***a*** *∈A*

and therefore had to solve an optimization problem over all actions.
We accelerated this by learning an approximate policy that “mimicked” this optimization,

***θ*** *[⋆]* *π* [=] [ arg max] **E** ***x*** *∼* *µ* � *Q* ( ***x***, ***π*** ( ***x*** ; ***θ*** ) ; ***θ*** *Q* ) � = . arg max *J* *µ* ( ***θ*** ; ***θ*** *Q* ) see eq. (12.75)
***θ*** ***θ***

where *µ* ( ***x*** ) *>* 0 was some exploration distribution that has full support and thus leads to the exploration of all states. The key idea was
that if we use a differentiable approximation *Q* and a differentiable
parameterization of policies, which is “rich enough”, then both optimizations are equivalent, and we can use the chain rule to obtain
gradient estimates of the second expression. This led us to the deterministic policy gradients (DDPG) algorithm. It turns out that there
is a very natural analogue to DDPG for model-based reinforcement
learning.

Instead of maximizing the Q-function directly, we use our finite-horizon
planning to estimate the immediate value of the policy within the next
*H* time steps and simply use the Q-function to approximate the terminal value (i.e., the tails of the infinite sum). Then, our objective
becomes,


�


*J* *µ*, *H* ( ***θ*** ) = . **E** ***x*** 0 *∼* *µ*, ***x*** 1: *H* *|* ***π*** ***θ***, *f*


*H* *−* 1
###### ∑ γ [τ] r τ + γ [H] Q ( x H, π ( x H ; θ ))
� *τ* = 0


(13.12)


This approach naturally extends to randomized policies using reparameterization gradients, which we have discussed in section 12.4.5.
For *H* = 0, this coincides with the DDPG objective! For larger time
horizons, the look-ahead facilitates taking into account the transition
model to plan what happens over the next time steps. This tends
to help dramatically in improving policies *much more rapidly* between
episodes. Instead of just gradually improving policies a little by slightly
adapting the policy to the learned value function estimates (as in modelfree RL), we use the model to anticipate the consequences of actions
multiple time steps ahead. This is at the heart of model-based reinforcement learning.

Essentially, we are using methods such as Q-learning and DDPG as
subroutines within the framework of model predictive control to do
much bigger steps in policy improvement than to slightly improve the
next picked action.

252 probabilistic artificial intelligence



*13.2* *Learning*

Thus far, we have considered known environments. That is, we as
sumed that the transition model *f* and the rewards *r* are known. In
reinforcement learning, *f* and *r* are, of course, not known. Instead, we

have to estimate them from data. This will also be crucial in our later

discussion of exploration in section 13.3 where we explore methods of
driving data collection to learn what we need to learn about the world.

First, let us revisit one of the key observations we made when we
first introduced the reinforcement problem. Namely, that the observed
transitions ***x*** *[′]* and rewards *r* are conditionally independent on the state
model-based approximate reinforcement learning 253

action pairs ( ***x***, ***a*** ) . [4] This is due to the Markovian structure of the 4 see eq. (11.3)
underlying Markov decision process.

This is the key observation that allows us to treat the estimation of
the dynamics and rewards as a simple regression problem (or a density estimation problem when the quantities are stochastic rather than
deterministic). More concretely, we can estimate the dynamics and rewards off-policy using the standard supervised learning techniques,
we discussed in earlier chapters, from a replay buffer,


*D* = *{* ( ***x*** *t*, ***a*** *t*, *r* *t*, ***x*** *t* + 1
���� ����
“input” “label”


) *}* *t* . (13.13)


Here, ***x*** *t* and ***a*** *t* are the “inputs”, and *r* *t* and ***x*** *t* + 1 are the “labels” of the
regression problem. Due to the conditional independence of the labels
given the inputs, we have independent label noise. This is the basic
assumption that we have been making throughout our discussion of
Bayesian learning.

The key difference to supervised learning is that the set of inputs depends on how we act. We will come back to this aspect of reinforcement learning in the next section on exploration. For now, recall that
we have used an arbitrary policy to collect some data, which we then
stored in a replay buffer, and which we now want to use to learn the
best-possible model of our environment.

*13.2.1* *Bayesian Learning*

In the following, we will discuss how we can use the techniques from
Bayesian learning, which we have seen in the first few chapters, to
learn the dynamics and reward models. We will focus on learning
the transition model *f*, however, learning the rewards model *r* is completely analogous. Moreover, we will focus on the setting where the
transition model is stochastic, that is, [5] 5 For learning, we can use for example
Gaussian processes or deep neural net***x*** *t* + 1 *∼* *f* ( ***x*** *t*, ***a*** *t* ; ***θ*** ) . (13.14) works.


254 probabilistic artificial intelligence



A first approach might be to obtain a point estimate for *f*, either
through maximum likelihood estimation (which we have seen to overfit easily) or through maximum a posteriori estimation. If *f* is represented as a deep neural network, we have already seen how to find the
MAP estimate of the weights in section 7.3.



first few chapters exploring. Notably, re


In the following, we will differentiate between the epistemic uncertainty and the aleatoric uncertainty. Recall from section 2.2 that epistemic uncertainty corresponds to our uncertainty about the model,

*p* ( *f* *| D* ), (13.16)

and aleatoric uncertainty corresponds to the uncertainty of the transitions in the underlying Markov decision process (i.e., the irreducible
noise),

*p* ( ***x*** *t* + 1 *|* *f*, ***x*** *t*, ***a*** *t* ) . (13.17)

Intuitively, Bayesian learning of dynamics models corresponds to learning a distribution over possible models *f* and *r*, where *f* and *r* charac


- section 2.2 for a description of epistemic and aleatoric uncertainty;

- chapter 4 for our use of uncertainty
estimates in the context of Gaussian

processes;

- chapter 7 for our use of uncertainty
estimates in the context of Bayesian
deep learning; and

- chapter 9 for our use of epistemic
uncertainty estimates to drive exploration.

model-based approximate reinforcement learning 255

terize a Markov decision process. This goes to show another benefit of
the model-based over the model-free approach to reinforcement learning. Namely, that it is much easier to encode prior knowledge (as a
Bayesian prior) about the transition and rewards model.



- eq. (7.19) for variational inference;

- eq. (7.22a) for Marko chain Monte
Carlo;

- eq. (7.27) for dropout regularization;
and

- eq. (7.28) for probabilistic ensembles.






In the context of planning, there is an important consequence of the
decomposition into epistemic and aleatoric uncertainty. In supervised
learning, we often conflated both notions. Consider the following: recall that the epistemic uncertainty corresponds to a distribution over
Markov decision processes *f*, whereas the aleatoric uncertainty corresponds to the randomness in the transitions within one such MDP *f* .
Crucially, this randomness in the transitions must be consistent within
a single MDP! That is, once we selected a single MDP for planning,
we want to disregard the epistemic uncertainty and solely focus on
the randomness of the transitions. Then, we want to average our plan
across the different realizations of *f* . This yields the following Monte
Carlo estimate of our planning reward *J* *H*,

*J* *H* ( ***a*** *t* : *t* + *H* *−* 1 )


*t* + *H* *−* 1
###### ∑ γ [τ] [−] [t] r ( x τ ( ϵ t [(] : [i] τ [)] − 1 [;] [ a] [t] [:] [τ] [−] [1] [,] [ f] [ (] [i] [)] [)] [,] [ a] [τ] [) +] [ γ] [H] [V] [(] [x] [t] [+] [H] [)]

*τ* = *t*

� �� �
= . *J* *H* ( ***a*** *t* : *t* + *H* *−* 1 ; *f* ( *i* ) )


*≈* [1]

*m*


*m*
###### ∑

*i* = 1


(13.19)


iid
where *f* [(] *[i]* [)] *∼* *p* ( *f* *| D* ) are independent samples of the transition

iid
model, ***ϵ*** *t* [(] : *[i]* *t* [)] + *H* *−* 1 *∼* *ϕ* are independent samples, and analogously to

256 probabilistic artificial intelligence

eq. (13.9),

***x*** *τ* ( ***ϵ*** *t* : *τ* *−* 1 ; ***a*** *t* : *τ* *−* 1, *f* )
= . *f* ( ***ϵ*** *τ* *−* 1 ; *f* ( . . . ; ( ***ϵ*** *t* + 1 ; *f* ( ***ϵ*** *t* ; ***x*** *t*, ***a*** *t* ), ***a*** *t* + 1 ), . . . ), ***a*** *τ* *−* 1 ) . (13.20)

Observe that the epistemic and aleatoric uncertainty are treated differently. Within a particular MDP *f*, we ensure that randomness (i.e., aleatoric uncertainty) is simulated consistently using our previous framework from our discussion of planning (see (13.10)). The Monte Carlo
samples of *f* take into account the epistemic uncertainty about the
transition model. Note that in our previous discussion of planning,
we assumed the Markov decision process *f* to be fixed. Essentially, we
use Monte Carlo trajectory sampling as a subroutine and average over
an “ensemble” of Markov decision processes.

Figure 13.4: Illustration of planning with
epistemic uncertainty and Monte Carlo
sampling. The agent considers *m* alternative “worlds”. Within each world, it
plans a sequence of actions over a finite time horizon. Then, the agent averages all optimal initial actions from
all worlds. Crucially, each world by itself is *consistent* . That is, its transition
model (i.e., the aleatoric uncertainty of
the model) is constant.

The same approach as we have seen in section 13.1.3 can be used to
“compile” these plans into a parametric policy that can be trained offline. [8] 8 In this case, we write *J* *H* ( *π* ) instead of
*J* *H* ( ***a*** *t* : *t* + *H* *−* 1 ) .
This leads us to a first greedily exploitative algorithm for model-based
reinforcement learning shown in alg. 13.9. This algorithm is purely
exploitative, as it greedily maximizes the expected reward with respect
to the transition model, taking into account epistemic uncertainty.


In the context of Gaussian process models, this algorithm is called
*probabilistic inference for learning control* (PILCO), [9] which was the first

demonstration of how sample efficient model-based reinforcement learning can be. As was mentioned in remark 13.5, PILCO uses moment
matching instead of Monte Carlo averaging.


9 Marc Deisenroth and Carl E Ras
mussen. Pilco: A model-based and dataefficient approach to policy search. In
*Proceedings of the 28th International Con-*
*ference on machine learning (ICML-11)*,
pages 465–472. Citeseer, 2011


In the context of neural networks, this algorithm is called *probabilistic*
*ensembles with trajectory sampling* (PETS), [10] which is one of the state- 10 Kurtland Chua, Roberto Calandra,
Rowan McAllister, and Sergey Levine.
Deep reinforcement learning in a handful of trials using probabilistic dynamics models. *Advances in neural information*
*processing systems*, 31, 2018

model-based approximate reinforcement learning 257

**Al** **g** **orithm 13.9:** Greed y ex p loitation for model-based RL

**1** start with (possibly empty) data *D* = ∅ and a prior *p* ( *f* ) = *p* ( *f* *| D* )

**2** `for` `several episodes` `do`

**3** plan a new policy *π* to (approximately) maximize,

max *π* **E** *f* *∼* *p* ( *·|D* ) [ *J* *H* ( *π* ; *f* )] (13.21)

**4** roll out policy *π* to collect more data

**5** update posterior *p* ( *f* *| D* )

Figure 13.5: Sample efficiency of modelfree and model-based reinforcement
learning. DDPG and SAC are shown as
constant (black) lines, because they take
an order of magnitude more time steps
before learning a good model. They also
compare the PETS algorithm (blue) to
deterministic ensembles (orange), where
the transition models are assumed to be

deterministic (or to have covariance 0).
Taken from “Deep reinforcement learning in a handful of trials using probabilistic dynamics models” (Chua et al.).

258 probabilistic artificial intelligence

of-the-art algorithms. PETS uses an ensemble of conditional Gaussian
distributions over weights, trajectory sampling for evaluating the performance and model predictive control for planning. Notably, PETS
does not explicitly explore. Exploration only happens due to the uncertainty in the model, which already drives exploration to some ex
tent.

In many settings, however, this incentive is not sufficient for exploration — especially when rewards are sparse.

*13.3* *Exploration*

Unlike in the setting of supervised learning, exploration is crucial for
reinforcement learning. We first encountered the exploration-exploitation
tradeoff in our discussion of Bayesian optimization, where we aimed
at maximizing an unknown function using noisy estimates in as little
time as possible. [11] Within the framework of Bayesian optimization, 11 see chapter 9
we used so-called acquisition functions for selecting the next point at
which to observe the unknown function. Observe that these acquisition functions are an analogue to policies in the setting of reinforcement learning. In particular, the policy that uses greedy exploitation
like we have seen in the previous section is analogous to simply picking the point that maximizes the expected value of the posterior distribution. In the context of Bayesian optimization, we have already
seen that this is insufficient for exploration and can easily get stuck in
locally optimal solutions. Thus, as reinforcement learning is a strict
generalization of Bayesian optimization, there is no reason why such
a strategy should be sufficient now.

Recall from our discussion of Bayesian optimization that we “solved”
this problem by using the epistemic uncertainty in our model of the
unknown function to pick the next point to explore. This is what we
will now explore in the context of reinforcement learning.

One simple strategy that we already investigated is the addition of
some *exploration noise* . In other words, one follows a greedy exploitative strategy, but every once in a while, one chooses a random action

model-based approximate reinforcement learning 259

(like in *ϵ* -greedy); or one adds additional noise to the selected actions
(known as Gaussian noise “dithering”). We have already seen that in
difficult exploration tasks, these strategies are not systematic enough.

Two other approaches that we have considered before are Thompson sampling [12] and the principle of “optimism in the face of uncer- 12 see section 9.3.4
tainty” [13] . 13 see section 9.2.1

*13.3.1* *Thompson Sampling*

Recall from section 9.3.4 that the idea behind Thompson sampling was
that the randomness in the realizations of *f* is already enough to drive
exploration. That is, instead of picking the action that performs best
on average across all realizations of *f*, Thompson sampling picks the
action that performs best for a *single realization* of *f* . The epistemic
uncertainty in the realizations of *f* leads to variance in the picked actions and provides an additional incentive for exploration. This yields
alg. 13.10 which is an immediate adaptation of greedy exploitation.

**Al** **g** **orithm 13.10:** Thom p son sam p lin g

**1** start with (possibly empty) data *D* = ∅ and a prior *p* ( *f* ) = *p* ( *f* *| D* )

**2** `for` `several episodes` `do`

**3** sample a model *f* *∼* *p* ( *· | D* )

**4** plan a new policy *π* to (approximately) maximize,

max *J* *H* ( *π* ; *f* ) (13.22)
*π*

**5** roll out policy *π* to collect more data

**6** update posterior *p* ( *f* *| D* )

*13.3.2* *Optimistic Exploration*

We have already seen in the context of multi-armed bandits and tabular reinforcement learning that optimism is a central pillar for exploration. But what would it mean to do optimistic exploration on
model-based reinforcement learning?

Let us consider a set *M* ( *D* ) of *plausible models* given some data *D* . Optimistic exploration would then optimize for the most advantageous
model among all models that are plausible given the seen data.

260 probabilistic artificial intelligence



When compared to greedy exploitation, instead of taking the optimal
step on average with respect to all realizations of the transition model
*f*, optimistic exploration takes the optimal step with respect to the
most optimistic model among all transition models that are consistent

with the data.

**Al** **g** **orithm 13.12:** O p timistic ex p loration

**1** start with (possibly empty) data *D* = ∅ and a prior *p* ( *f* ) = *p* ( *f* *| D* )

**2** `for` `several episodes` `do`

**3** plan a new policy *π* to (approximately) maximize,

max max (13.24)
*π* *f* *∈* *M* ( *D* ) *[J]* *[H]* [(] *[π]* [;] *[ f]* [ )]

**4** roll out policy *π* to collect more data

**5** update posterior *p* ( *f* *| D* )

In general, the joint maximization over *π* and *f* is very difficult. Yet,
remarkably, it turns out that this complex optimization can be reduced
to standard model-based reinforcement learning with a fixed model.

The idea is to consider an agent that can control its “luck”. In other
words, we assume that the agent believes it can control the outcome of
its actions — or rather choose which of the plausible dynamics it follows. The “luck” of the agent can be considered as additional decision
variables. Consider the optimization problem,

*π* *t* = . arg max max (13.25)
*π* ***η*** ( *·* ) *∈* [ *−* 1,1 ] *[d]* *[ J]* *[H]* [(] *[π]* [; ˜] *[f]* [ )]

where

*f* ˜ ( ***x***, ***a*** ) = . ***µ*** *t* *−* 1 ( ***x***, ***a*** ) + *β* *t* *−* 1 ***Σ*** *t* *−* 1 ( ***x***, ***a*** ) ***η*** ( ***x***, ***a*** ) . (13.26)

model-based approximate reinforcement learning 261

Here the decision variables ***η*** control the variance of an action. That
is, within the confidence bounds of the transition model, the agent can
freely choose the state that is reached by playing an action ***a*** from state
***x*** . Essentially, this corresponds to maximizing expected reward in an
augmented (optimistic) MDP with dynamics *f* [˜] and with a larger action
space that also includes the decision variables ***η*** . This is a known MDP
for which we can use our toolbox for planning which we developed in
section 13.1.


The algorithm that maximizes expected reward in this augmented
MDP is called *hallucinated upper confidence reinforcement learning* (HUCRL). [14] An illustration of the algorithm is given in fig. 13.6.


14 Sebastian Curi, Felix Berkenkamp, and
Andreas Krause. Efficient model-based
reinforcement learning through optimistic policy search and planning. *arXiv*
*preprint arXiv:2006.08684*, 2020

Figure 13.6: Illustration of H-UCRL in a
one-dimensional state space. The agent
“hallucinates” that it takes the black trajectory when, in reality, the outcomes of
its actions are as shown in blue. The
agent can hallucinate to land anywhere
within the gray confidence regions (i.e.,
the epistemic uncertainty in the model)
using the luck decision variables ***η*** . This
allows agents to discover long sequences
of actions leading to sparse rewards.





*t*

Clearly, as more data is collected, the confidence bounds shrink and
the optimistic policy rewards converge to the actual rewards. Yet, crucially, we only collect data in regions of the state-action space that are
more promising than the regions that we have already explored. That
is, we only collect data in the most promising regions. Intuitively, the
agent has the tendency to believe that it can achieve much more than
it actually can.


Optimistic exploration yields the strongest effects for hard exploration
tasks, for example, settings with large penalties associated with performing certain actions and settings with sparse rewards. [15] In those 15 Action penalties are often used to dis
settings, most other strategies (i.e., those that are not exploratory enough), courage the agent from exhibiting un
wanted behavior. However, increasing

learn not to act at all. However, even in settings of “ordinary rewards”, the action penalty increases the difficulty
optimistic exploration often learns good policies faster. of exploration. Therefore, optimistic ex
ploration is especially useful in settings
where we want to practically disallow
many actions by attributing large penalties to them.

262 probabilistic artificial intelligence

*13.3.3* *Constrained Exploration*

Besides making exploration more efficient, another use of uncertainty
is to make exploration more safe. Today, reinforcement learning is still
far away from being deployed directly to the real world. In practice,
reinforcement learning is almost always used together with a simulator, in which the agent can “safely” train and explore. Yet, in many
domains, it is not possible to simulate the training process, either because we are lacking a perfect model of the environment, or because
simulating such a model is too computationally inefficient. This is
where we can make use of uncertainty estimates of our model to avoid

unsafe states.

Let us denote by *X* unsafe the subset of unsafe states of our state space
*X* . A natural idea is to perform planning using confidence bounds
of our epistemic uncertainty. This allows us to pessimistically forecast
the plausible consequences of our actions, given what we have already
learned about the transition model. As we collect more data, the con
fidence bounds will shrink, requiring us to be less conservative over
time. This idea is also at the heart of fields like *robust control* .




Figure 13.7: Illustration of planning with
confidence bounds. The unsafe set of
states is shown as the red region. The
agent starts at the position denoted by
the black dot and plans a sequence of
actions. The confidence bounds on the
subsets of states that are reached by this
action sequence are shown in gray.


A general formalism for planning under constraints is the notion of
*constrained Markov decision processes* . [16] Given the true dynamics *f* *[⋆]*, 16 Eitan Altman. *Constrained Markov de-*
planning in constrained MDPs amounts to the following optimization *cision processes: stochastic modeling* . Routledge, 1999
problem:


max *π* *J* *µ* ( *π* ; *f* *[⋆]* ) = . **E** ***x*** 0 *∼* *µ*, ***x*** 1:∞ *|* *π*, *f*


∞
###### ∑ γ [t] r t
� *t* = 0 �


(13.27a)


subject to *J* *µ* *[c]* *[i]* ( *π* ; *f* *[⋆]* ) *≤* *δ* *i* *∀* *i* (13.27b)

where *µ* is a distribution over the initial state and


*J* *µ* *[c]* *[i]* ( *π* ; *f* ) = . **E** ***x*** 0 *∼* *µ*, ***x*** 1:∞ *|* *π*, *f*


∞
###### ∑ γ [t] c i ( x t )
� *t* = 0 �


(13.28)


are expected discounted costs with respect to a cost function *c* *i* : *X →*
**R** *≥* 0 . Observe that for the cost function *c* ( ***x*** ) = . **1** *{* ***x*** *∈X* unsafe *}*, the
value *J* *µ* *[c]* [(] *[π]* [;] *[ f]* *[ ⋆]* [)] [ can be interpreted as an upper bound on the (dis-]
counted) probability of visiting unsafe states, [17] and hence, the con- 17 This follows from a simple union
straint *J* *µ* *[c]* [(] *[π]* [;] *[ f]* *[ ⋆]* [)] *[ ≤]* *[δ]* [ bounds the probability of visiting an unsafe state] bound (1.1).
when following policy *π* .

The *augmented Lagrangian method* can be used to solve the optimization
problem of eq. (13.27). [18] Thereby, one solves 18 For an introduction, read chapter 17
of “Numerical optimization” (Wright
et al.).
###### max π min λ ≥ 0 J µ ( π ; f ) − ∑ i λ i ( J µ [c] [i] ( π ; f ) − δ i ) (13.29)

model-based approximate reinforcement learning 263


(13.30)
*−* ∞ otherwise.


= max

*π*






 *−* *J* *µ* ∞ ( *π* ; *f* ) ifotherwise. *π* is feasible




Observe that if *π* is feasible, then *J* *µ* *[c]* *[i]* ( *π* ; *f* ) *≤* *δ* and so the minimum
over ***λ*** is satisfied if ***λ*** = 0. Conversely, if *π* is infeasible, at least one
*λ* *i* can be made arbitrarily large to solve the optimization problem. In
practice, an additional penalty term is added to smooth the objective
when transitioning between feasible and infeasible policies.

Note that solving constrained optimization problems such as eq. (13.27)
yields an optimal safe policy. However, it is not ensured that constraints are not violated during the search for the optimal policy. Generally, exterior penalty methods such as the augmented Lagrangian
method allow for generating infeasible points during the search, and
are therefore not suitable when constraints have to be strictly enforced
at all times. Thus, this method is more applicable in the episodic setting (e.g. when an agent is first “trained” in a simulated environment
and then “deployed” to the actual task) rather than in the continuous
setting where the agent has to operate in the “real world” from the
beginning and cannot easily be reset.



So far we have assumed knowledge of the true dynamics *f* *[⋆]* . If we
do not know the true dynamics but instead have access to a set of
plausible models *M* ( *D* ) given data *D* (cf. section 13.3.2) such that *f* *[⋆]* *∈*
*M* ( *D* ), then a natural strategy is to be optimistic with respect to future
rewards and pessimistic with respect to future constraint violations.

264 probabilistic artificial intelligence

More specifically, we solve the optimization problem,

max *π* *f* *∈* max *M* ( *D* ) *J* *µ* ( *π* ; *f* ) (13.32a) *optimistic*

subject to max *∀* *i* . (13.32b) *pessimistic*
*f* *∈* *M* ( *D* ) *[J]* *[µ][c]* *[i]* [ (] *[π]* [;] *[ f]* [ )] *[ ≤]* *[δ]* *[i]*

Intuitively, jointly maximizing *J* *µ* ( *π* ; *f* ) with respect to *π* and (plausible) *f* can lead the agent to try behaviors with potentially high reward
due to *optimism* (i.e., the agent “dreams” about potential outcomes).
On the other hand, being *pessimistic* with respect to constraint violations enforces the safety constraints (i.e., the agent has “nightmares”
about potential outcomes).

If the policy values *J* *µ* ( *π* ; *f* ) and *J* *µ* *[c]* *[i]* ( *π* ; *f* ) are modeled as a distribution (e.g., using Bayesian neural networks), then the inner maximizations over plausible dynamics can be approximated using samples
from the posterior distributions. Thus, the augmented Lagrangian
method can also be used to solve the general optimization problem
of eq. (13.32). The resulting algorithm is known as *Lagrangian model-*
*based agent* (LAMBDA). [19] 19 Yarden As, Ilnura Usmanova, Sebastian Curi, and Andreas Krause.
Constrained policy optimization via
*13.3.4* *Safe Exploration* bayesian world models. *arXiv preprint*
*arXiv:2201.09802*, 2022

As noted in the previous section, in many settings we do not only
want to eventually find a safe policy, but we also want to ensure that
we act safely while searching for an optimal policy. To this end, recall the general approach outlined in the beginning of the previous
section wherein we plan using (pessimistic) confidence bounds of the
plausible consequences of our actions.


One key challenge of this approach is that we need to forecast plausible trajectories. The confidence bounds of such trajectories cannot be
nicely represented anymore. One approach is to over-approximate the
confidence bounds over reachable states from trajectories.

**Theorem 13.14.** *For conditional Gaussian dynamics, the reachable states of*
*trajectories can be over-approximated with high probability.* [20]

Another key challenge is that actions might have consequences that
exceed the time horizon used for planning. In other words, by performing an action now, our agent might put himself into a state that
it is not yet unsafe, but out of which it cannot escape and which will
eventually lead to an unsafe state. You may think of a car driving towards a wall. When a crash with the wall is designated as an unsafe
state, there are quite a few states before the crash at which it is already
impossible to avoid the crash. Thus, looking ahead a finite number of
steps is not sufficient to prevent entering unsafe states.


20 Torsten Koller, Felix Berkenkamp,
Matteo Turchetta, and Andreas Krause.
Learning-based model predictive control
for safe exploration. In *2018 IEEE confer-*
*ence on decision and control (CDC)*, pages
6059–6066. IEEE, 2018

unsafe trajectory

Figure 13.8: Illustration of long-term
consequence when planning a finite
number of steps. Green dots are to denote safe states and the red dot is to denote an unsafe state. After performing
the first action, the agent is still able to
return to the previous state. Yet, after
reaching the third state, the agent is already guaranteed to end in an unsafe
state. When using only a finite horizon
of *H* = 2 for planning, the agent might
make this transition regardless.

model-based approximate reinforcement learning 265


It turns out that it is still possible to use the epistemic uncertainty
about the model and short-term plausible behaviors to make guarantees about certain long-term consequences. An idea is to also consider
a set of safe states *X* safe alongside the set of unsafe states *X* unsafe, for
which our agent knows how to stay inside (i.e., remain safe). In other
words, for states ***x*** *∈X* safe, we know that our agent can always behave
in such a way that it will not reach an unsafe state. [21] An illustration 21 In the example of driving a car, the set

of this approach is shown in fig. 13.9. of safe states corresponds to those states

where we know that we can still safely
break before hitting the wall.


of this approach is shown in fig. 13.9.


The problem is that this set of safe states might be very conservative.
That is to say, it is likely that rewards are mostly attained *outside* of
the set of safe states. The key idea is to plan two sequences of actions,
instead of only one. “Plan A” (the *performance plan* ) is planned with
the objective to solve the task. [22] “Plan B” (the *safety plan* ) is planned 22 that is, attain maximum reward in the
with the objective to return to the set of safe states *X* safe . In addition, sense that we have discussed until now
we enforce that both plans must agree on the first action to be played.


Figure 13.9: Illustration of long-term
consequence when planning a finite
number of steps. The unsafe set of states
is shown in red and the safe set of states
is shown in blue. The confidence intervals corresponding to actions of the
performance plan and safety plan are
shown in orange and green, respectively.



Under the assumption that the confidence bounds are conservative
estimates, this guarantees that after playing this action, our agent will

still be in a state of which it can return to the set of safe states. In this

way, we can gradually increase the set of states that are safe to explore.
It can be shown that under suitable conditions, returning to the safe
set can be guaranteed. [23] 23 Torsten Koller, Felix Berkenkamp,
Matteo Turchetta, and Andreas Krause.
Learning-based model predictive control

*13.3.5* *Safety Filters* for safe exploration. In *2018 IEEE confer-*

*ence on decision and control (CDC)*, pages

It is also possible to (slightly) adjust any potentially unsafe policy *π* 6059–6066. IEEE, 2018


*13.3.5* *Safety Filters*


It is also possible to (slightly) adjust any potentially unsafe policy *π*
to obtain a policy ˆ *π* which avoids entering unsafe states with high
probability.


Following our interpretation of constrained policy optimization in terms
of optimism and pessimism from section 13.3.3, to pessimistically evaluate the safety of a policy with respect to a cost function *c* given a set

266 probabilistic artificial intelligence

of plausible models *M* ( *D* ), we can use

*C* *[π]* ( ***x*** ) = . max *δ* ***x*** [(] *[π]* [;] *[ f]* [ )] (13.33) *δ* ***x*** is the point density at ***x***
*f* *∈* *M* ( *D* ) *[J]* *[c]*

where the initial-state distribution *δ* ***x*** is to denote that the initial state

is ***x*** .

Observe that eq. (13.33) permits a reparameterization in terms of additional decision variables ***η*** which is analogous to our discussion in
section 13.3.2. More specifically, we have

*C* *[π]* ( ***x*** ) = max *J* *δ* *[c]* ***x*** [(] *[π]* [; ˜] *[f]* [ )] (13.34)
***η***

where *f* [˜] are the adjusted dynamics (13.26) which are based on the
“luck” variables ***η*** . In the context of estimating costs which we aim to
minimize (as opposed to rewards which we aim to maximize), ***η*** can
be interpreted as the “bad luck” of the agent.

When our only objective is to act safely, that is, we only aim to minimize cost and are indifferent to rewards, then this reparameterization
allows us to find a “maximally safe” policy,

*π* safe = . arg min **E** ***x*** *∼* *µ* [ *C* *[π]* ( ***x*** )] = arg min max *J* *µ* *[c]* [(] *[π]* [; ˜] *[f]* [ )] [.] (13.35)
*π* *π* ***η***

Under some conditions it can be shown that *π* safe = *π* ***x*** for any ***x***
where *π* ***x*** = . arg min *π* *C* *π* ( ***x*** ) . On its own, the policy *π* safe is rather
useless for exploring the state space. In particular, when in a state that
we already deem safe, following policy *π* safe the agent aims simply to
“stay” within the set of safe states which means that it has no incentive
to explore.

Instead, one can interpret *π* safe as a “backup” policy in case we realize
upon exploring that a certain trajectory is too dangerous akin to our
notion of the “safety plan” in section 13.3.4. That is, given any (potentially explorative and dangerous) policy *π*, we can find the adjusted
policy,

ˆ .
*π* ( ***x*** ) = arg min *d* ( *π* ( ***x*** ), ***a*** ) (13.36a)
***a*** *∈A*

subject to max *C* *[π]* [safe] ( *f* [˜] ( ***x***, ***a*** )) *≤* *δ* (13.36b)
***η***

for some metric *d* ( *·*, *·* ) on *A* . The constraint ensures that the pessimistic
next state *f* [˜] ( ***x***, ***a*** ) is recoverable by following policy *π* safe . In this way,


(13.37)
*π* safe ( ***x*** ) if *C* *[π]* [safe] ( ***x*** ) *>* *δ*


*π* ˜ ( ***x*** ) = .






*π* ˆ ( ***x*** ) if *C* *[π]* [safe] ( ***x*** ) *≤* *δ*

 *π* safe ( ***x*** ) if *C* *[π]* [safe] ( ***x*** ) *>* *δ*



model-based approximate reinforcement learning 267

is the policy “closest” to *π* which is *δ* -safe with respect to the pessimistic cost estimates *C* *[π]* [safe] . [24] [,] [25] This is also called a *safety filter* . Simi- 24 The policy ˜ *π* is required as, a priori,

lar approaches using backup policies also allow safe exploration in the it is not guaranteed that the state ***x*** *t* will

satisfy *C* *[π]* [safe] ( ***x*** *t* ) *≤* *δ* for all *t* unless ˆ *π* is

model-free setting. [26] replanned after every step.



25 Sebastian Curi, Armin Lederer, Sandra
Hirche, and Andreas Krause. Safe reinforcement learning via confidence-based
filters. *arXiv preprint arXiv:2207.01337*,

2022

26 Bhavya Sukhija, Matteo Turchetta,
David Lindner, Andreas Krause, Sebastian Trimpe, and Dominik Baumann.
Scalable safe exploration for global optimization of dynamical systems. *arXiv*
*preprint arXiv:2201.09562*, 2022

### *A* *Solutions*

*Fundamentals*

**Solution to exercise 1.5 (Properties of probability).**

1. By the third axiom and *B* = *A* *∪* ( *B* *\* *A* ),

**P** ( *B* ) = **P** ( *A* ) + **P** ( *B* *\* *A* ) .

Noting from the first axiom that **P** ( *B* *\* *A* ) *≥* 0 completes the proof.
2. By the second axiom,

**P** � *A* *∪* *A* � = **P** ( Ω ) = 1

and by the third axiom,


**P** � *A* *∪* *A* � = **P** ( *A* ) + **P** �


*A* .
�


Reorganizing the equations completes the proof.
3. Define the countable sequence of events


� *A* *j*

*j* = 1


*A* *i* *[′]* = . *A* *i* *\*





*i* *−* 1



�
 *j* = 1


.


Note that the sequence of events *{* *A* *i* *[′]* *[}]* *[i]* [ is disjoint. Thus, we have]
by the third axiom and then using (1) that


�


∞
� *A* *i* *[′]*
� *i* = 1


∞
�
� *i* =


�


**P**


∞
� *A* *i*
� *i* = 1


= **P**


∞ ∞
###### = ∑ P � A i [′] � ≤ ∑ P ( A i ) .

*i* = 1 *i* = 1


**Solution to exercise 1.21 (Random walks on graphs).** We show that
any vertex *v* is visited eventually with probability 1.

We denote by *w* *→* *v* the event that the random walk starting at vertex
*w* visits the vertex *v* eventually, we denote by Γ ( *w* ) the neighborhood

270 probabilistic artificial intelligence

of *w*, and we write deg ( *w* ) = . *|* Γ ( *w* ) *|* . We have,
###### P ( w → v ) = ∑ P �the random walk first visits w [′] [�] · P � w [′] → v � using the law of total probability (1.15)

*w* *[′]* *∈* Γ ( *w* )

1
###### = ∑ P � w [′] → v �. using that the random walk moves to a
deg ( *w* ) *w* *[′]* *∈* Γ ( *w* ) neighbor uniformly at random

Take *u* to be the vertex such that **P** ( *u* *→* *v* ) is minimized. Then,


1
###### P ( u → v ) = ∑ P � u [′] → v �
deg ( *u* ) *u* *[′]* *∈* Γ ( *u* ) � �� �

*≥* **P** ( *u* *→* *v* )


*≥* **P** ( *u* *→* *v* ) .


That is, for all neighbors *u* *[′]* of *u*, **P** ( *u* *→* *v* ) = **P** ( *u* *[′]* *→* *v* ) . Using that
the graph is connected and finite, we conclude **P** ( *u* *→* *v* ) = **P** ( *w* *→* *v* )
for any vertex *w* . Finally, note that **P** ( *v* *→* *v* ) = 1, and hence, the
random walk starting at any vertex *u* visits the vertex *v* eventually
with probability 1.

**Solution to exercise 1.28 (Law of total expectation).** Let **1** *{* *A* *i* *}* be the
indicator random variable for the event *A* *i* . Then,

**E** [ **X** *·* **1** *{* *A* *i* *}* ] = **E** [ **E** [ **X** *·* **1** *{* *A* *i* *} |* **1** *{* *A* *i* *}* ]] by the tower rule (1.33)


= **E** [ **X** *·* **1** *{* *A* *i* *} |* **1** *{* *A* *i* *}* = 1 ] *·* **P** ( **1** *{* *A* *i* *}* = 1 )

+ 0 *·* **P** ( **1** *{* *A* *i* *}* = 0 )


expanding the outer expectation


= **E** [ **X** *|* *A* *i* ] *·* **P** ( *A* *i* ) . the event **1** *{* *A* *i* *}* = 1 is equivalent to *A* *i*

Summing up for all *i*, the left-hand side becomes

*k*
###### E [ X ] = ∑ E [ X · 1 { A i } ] .

*i* = 1

**Solution to exercise 1.30 (Covariance matrices are positive semi-def-**
**inite).** Let ***Σ*** = . Var [ **X** ] be a covariance matrix of the random vector **X**
and fix any ***z*** *∈* **R** *[n]* . Then,

***z*** *[⊤]* ***Σz*** = ***z*** *[⊤]* **E** � ( **X** *−* **E** [ **X** ])( **X** *−* **E** [ **X** ]) *[⊤]* [�] ***z*** using the definitiion of variance (1.44)

= **E** � ***z*** *[⊤]* ( **X** *−* **E** [ **X** ])( **X** *−* **E** [ **X** ]) *[⊤]* ***z*** �. using linearity of expectation (1.24)

Define the random variable *Z* = . ***z*** *⊤* ( **X** *−* **E** [ **X** ]) . Then,

= **E** *Z* [2] [�] *≥* 0.
�

**Solution to exercise 1.35 (Bayes’ rule).** Let us start by defining some
events that we will reason about later. For ease of writing, let us call
the person in question X.

*D* = X has the disease,

solutions 271


*P* = The test shows a positive response.

Now we can translate the information in the question to formal statements in terms of *D* and *P*,

**P** ( *D* ) = 10 *[−]* [4] the disease is rare


**P** ( *P* *|* *D* ) = **P** � *P* *|* *D* � = 0.99. the test is accurate


We want to determine **P** ( *D* *|* *P* ) . One can find this probability by using
Bayes’ rule (1.58),

**P** ( *D* *|* *P* ) = **[P]** [(] *[P]* *[|]* *[ D]* [)] *[ ·]* **[ P]** [(] *[D]* [)] .

**P** ( *P* )

From the quantities above, we have everything except for **P** ( *P* ) . This,
however, we can compute using the law of total probability,


**P** ( *P* ) = **P** ( *P* *|* *D* ) *·* **P** ( *D* ) + **P** � *P* *|* *D* � *·* **P** �

= 0.99 *·* 10 *[−]* [4] + 0.01 *·* ( 1 *−* 10 *[−]* [4] )

= 0.010098.


*D* �


Hence, **P** ( *D* *|* *P* ) = 0.99 *·* 10 *[−]* [4] /0.010098 *≈* 0.0098 = 0.98%.

**Solution to exercise 1.40 (Sample variance).** Using that *X* is zeromean, we have that


**E**
�


1

*X* [2] *n* � = Var� *X* *n* � = *n* [Var] [[] *[X]* []] and


1

**E**

*n*
�


*n*
*x* [(] *[i]* [)] [2]
###### ∑
*i* = 1 �


= [1]

*n*


*n*
###### ∑ E � x [(] [i] [)] [2] [�] = Var [ X ] .

*i* = 1


Thus,


1

**E**

*n*
�


*n*
*x* [(] *[i]* [)] [2]
###### ∑
*i* = 1 �


*n*
**E** *S* *n* [2] =
� � *n* *−* 1


�


*−* **E**
�


*X* [2] *n*
� [�]


= Var [ *X* ] .


**Solution to exercise 1.44 (Simple concentration inequalities).**

1. W.l.o.g. we assume that *X* is continuous. We have


∞
**E** *X* =
� 0


= *x* *·* *p* ( *x* ) *dx* using the definition of expectation

0 (1.23b)
*ϵ* ∞ ∞
= + *x* *·* *p* ( *x* ) *dx* *≥* *ϵ* *·* *p* ( *x* ) *dx* .
� 0 *[x]* *[ ·]* *[ p]* [(] *[x]* [)] *[ dx]* � *ϵ* � *ϵ*


∞

+
� *ϵ*



*·* *p* ( *x* ) *dx*
*ϵ*
� �� �
**P** ( *X* *≥* *ϵ* )


0 *[x]* *[ ·]* *[ p]* [(] *[x]* [)] *[ dx]*
� �� �
*≥* 0


∞ ∞

*x* *·* *p* ( *x* ) *dx* *≥* *ϵ*
*ϵ* � *ϵ*


.


. 2
2. Consider the non-negative random variable *Y* = ( *X* *−* **E** *X* ) . We

have

**P** ( *|* *X* *−* **E** *X* *| ≥* *ϵ* ) = **P** ( *X* *−* **E** *X* ) [2] *≥* *ϵ* [2] [�]
�

2
� ( *X* *−* **E** *X* ) �
*≤* **[E]** using Markov’s inequality (1.81)

*ϵ* [2]

= [Var] *[X]* . using the definition of variance (1.44)

*ϵ* [2]

272 probabilistic artificial intelligence

**Solution to exercise 1.46 (Weak law of large numbers).** Fix any *ϵ* *>*
0. Applying Chebyshev’s inequality and noting that **E** *X* *n* = **E** *X*, we

obtain

**P** ��� *X* *n* *−* **E** *X* �� *≥* *ϵ* � *≤* Var *X* *n* .

*ϵ* [2]

We further have for the variance of the sample mean that


�

###### ∑ n Var [ X i ] = [Var] [X] .

*i* = 1 *n*


Var *X* *n* = Var


1

*n*
�


*n*
###### ∑ X i

*i* = 1


= [1]

*n* [2]


Thus,


Var *X*
*n* lim *→* ∞ **[P]** ��� *X* *n* *−* **E** *X* �� *≥* *ϵ* � *≤* *n* lim *→* ∞ *ϵ* [2] *n* [=] [ 0]


**P**
which is precisely the definition of *X* *n* *→* **E** *X* .

**Solution to exercise 1.56 (Directional derivatives).** Fix a *λ* *>* 0. Using a first-order expansion, we have

*f* ( ***x*** + *λ* ***d*** ) = *f* ( ***x*** ) + *λ* ***∇*** *f* ( ***x*** ) *[⊤]* ***d*** + *o* ( *λ* *∥* ***d*** *∥* 2 ) .

Dividing by *λ* yields,


***d*** ) *−* *f* ( ***x*** ) = ***∇*** *f* ( ***x*** ) *[⊤]* ***d*** + *[o]* [(] *[λ]* *[∥]* ***[d]*** *[∥]* [2] [)]

*λ* *λ*


*f* ( ***x*** + *λ* ***d*** ) *−* *f* ( ***x*** )


*λ*
� �� �
*→* 0


.


Taking the limit *λ* *→* 0 gives the desired result.

**Solution to exercise 1.66 (Product of Gaussian PDFs).** We need to

find ***µ*** *∈* **R** *[n]* and ***Σ*** *∈* **R** *[n]* *[×]* *[n]* such that

( ***x*** *−* ***µ*** ) *[⊤]* ***Σ*** *[−]* [1] ( ***x*** *−* ***µ*** ) ∝ ( ***x*** *−* ***µ*** 1 ) *[⊤]* ***Σ*** *[−]* 1 [1] [(] ***[x]*** *[ −]* ***[µ]*** [1] [) + (] ***[x]*** *[ −]* ***[µ]*** [2] [)] *[⊤]* ***[Σ]*** *[−]* 2 [1] [(] ***[x]*** *[ −]* ***[µ]*** [2] [)] [.]
(A.1)

for any ***x*** *∈* **R** *[n]* .

The left-hand side of eq. (A.1) is equal to

***x*** *[⊤]* ***Σ*** *[−]* [1] ***x*** *−* 2 ***x*** *[⊤]* ***Σ*** *[−]* [1] ***µ*** + ***µ*** *[⊤]* ***Σ*** *[−]* [1] ***µ*** .

The right-hand side of eq. (A.1) is equal to

***x*** *[⊤]* ***Σ*** *[−]* [1] *−* 2 ***x*** *[⊤]* ***Σ*** *[−]* [1]
1 ***[x]*** [ +] ***[ x]*** *[⊤]* ***[Σ]*** *[−]* 2 [1] ***[x]*** 1 ***[µ]*** [1] [ +] ***[ x]*** *[⊤]* ***[Σ]*** *[−]* 2 [1] ***[µ]*** [2]
� � � �

+ ***µ*** 1 *[⊤]* ***[Σ]*** *[−]* 1 [1] ***[µ]*** [1] [ +] ***[ µ]*** 2 *[⊤]* ***[Σ]*** *[−]* 2 [1] ***[µ]*** [2]
� �

= ***x*** *[⊤]* [�] ***Σ*** *[−]* 1 [1] + ***Σ*** *[−]* 2 [1] ***x*** *−* 2 ***x*** *[⊤]* [�] ***Σ*** *[−]* 1 [1] ***[µ]*** [1] [ +] ***[ Σ]*** *[−]* 2 [1] ***[µ]*** [2]
� �

+ ***µ*** 1 *[⊤]* ***[Σ]*** *[−]* 1 [1] ***[µ]*** [1] [ +] ***[ µ]*** 2 *[⊤]* ***[Σ]*** *[−]* 2 [1] ***[µ]*** [2] .
� �

We observe that both sides are equal if

***Σ*** *[−]* [1] = ***Σ*** *[−]* 1 [1] + ***Σ*** *[−]* 2 [1] and ***Σ*** *[−]* [1] ***µ*** = ***Σ*** *[−]* 1 [1] ***[µ]*** [1] [ +] ***[ Σ]*** *[−]* 2 [1] ***[µ]*** [2] [.]

solutions 273


**Solution to exercise 1.70 (Marginal / conditional distribution of a**
**Gaussian).** Let ***x*** *∼* **X** . The joint distribution can be expressed as

*p* ( ***x*** ) = *p* ( ***x*** *A*, ***x*** *B* )


� [] 


*⊤*
***Λ*** *AA* ***Λ*** *AB* ***x*** *A* *−* ***µ*** *A*
� � ***Λ*** *BA* ***Λ*** *BB* �� ***x*** *B* *−* ***µ*** *B*


= [1]

*Z* [exp]


= [1]





*−* [1]
 2


2


***x*** *A* *−* ***µ*** *A*
� ***x*** *B* *−* ***µ*** *B*


where *Z* denotes the normalizing constant. To simplify the notation,

we write
� ***∆∆*** *AB* � = . � ***xx*** *AB* *− −* ***µµ*** *BA* � .


= .
�


***x*** *A* *−* ***µ*** *A*
� ***x*** *B* *−* ***µ*** *B*


�


.


1. We obtain

*p* ( ***x*** *A* )


***∆*** *A*

***∆*** *B*
��


*⊤*
***Λ*** *AA* ***Λ*** *AB*

***Λ*** *BA* ***Λ*** *BB*
� �


*d* ***x*** *B* using the sum rule (1.10)

� [] 


= [1]

*Z*


exp
�


2





*−* [1]
 2


***∆*** *A*

***∆*** *B*
�



[1] *−* [1]

*Z* [exp] � 2


= [1]


2


***∆*** *[⊤]* *A* [(] ***[Λ]*** *[AA]* *[−]* ***[Λ]*** *[AB]* ***[Λ]*** *[−]* *BB* [1] ***[Λ]*** *[BA]* [)] ***[∆]*** *[A]*
� � [�]


using the first hint



*·* exp *−* [1] ( ***∆*** *B* + ***Λ*** *[−]* *BB* [1] ***[Λ]*** *[BA]* ***[∆]*** *[A]* [)] *[⊤]* ***[Λ]*** *[BB]* [(] ***[∆]*** *[B]* [ +] ***[ Λ]*** *[−]* *BB* [1] ***[Λ]*** *[BA]* ***[∆]*** *[A]* [)] *d* ***x*** *B* .
� � 2 � � [�]


Observe that the integrand is an unnormalized Gaussian PDF, and


hence, the integral evaluates to
�


det 2 *π* ***Λ*** *[−]* *BB* [1] (so is constant with
� �


respect to ***x*** *A* ). We can therefore simplify to



[1] *−* [1]

*Z* *[′]* [ exp] � 2

[1] *−* [1]

*Z* *[′]* [ exp] � 2


= [1]


2


= [1]


2


***∆*** *[⊤]* *A* [(] ***[Λ]*** *[AA]* *[−]* ***[Λ]*** *[AB]* ***[Λ]*** *[−]* *BB* [1] ***[Λ]*** *[BA]* [)] ***[∆]*** *[A]*
� � [�]

� ***∆*** *[⊤]* *A* ***[Σ]*** *[−]* *AA* [1] ***[∆]*** *[A]* � [�] . using the second hint


2. We obtain

*p* ( ***x*** *B* *|* ***x*** *A* )

= *[p]* [(] ***[x]*** *[A]* [,] ***[ x]*** *[B]* [)] using the definition of conditional

*p* ( ***x*** *A* ) distributions (1.13)


***∆*** *A*

***∆*** *B*
��


noting that *p* ( ***x*** *A* ) is constant with

� []  respect to ***x*** *B*


= [1]

*Z* *[′]* [ exp]


= [1]





*−* [1]
 2


2


*⊤*

***∆*** *A* ***Λ*** *AA* ***Λ*** *AB*

***∆*** *B* ***Λ*** *BA* ***Λ*** *BB*
� � �



[1] *−* [1]

*Z* *[′]* [ exp] � 2


= [1]


2


***∆*** *[⊤]* *A* [(] ***[Λ]*** *[AA]* *[−]* ***[Λ]*** *[AB]* ***[Λ]*** *[−]* *BB* [1] ***[Λ]*** *[BA]* [)] ***[∆]*** *[A]*
� � [�]


***∆*** *A* [(] ***[Λ]*** *[AA]* *[−]* ***[Λ]*** *[AB]* ***[Λ]*** *BB* ***[Λ]*** *[BA]* [)] ***[∆]*** *[A]* using the first hint

( ***∆*** *B* + ***Λ*** *[−]* *BB* [1] ***[Λ]*** *[BA]* ***[∆]*** *[A]* [)] *[⊤]* ***[Λ]*** *[BB]* [(] ***[∆]*** *[B]* [ +] ***[ Λ]*** *[−]* *BB* [1] ***[Λ]*** *[BA]* ***[∆]*** *[A]* [)]
� � [�]



*·* exp *−* [1]
� 2


2



[1] *−* [1]

*Z* *[′′]* [ exp] � 2


= [1]


2


( ***∆*** *B* + ***Λ*** *[−]* *BB* [1] ***[Λ]*** *[BA]* ***[∆]*** *[A]* [)] *[⊤]* ***[Λ]*** *[BB]* [(] ***[∆]*** *[B]* [ +] ***[ Λ]*** *[−]* *BB* [1] ***[Λ]*** *[BA]* ***[∆]*** *[A]* [)] . observing that the first exponential is
� � [�]
constant with respect to ***x*** *B*

274 probabilistic artificial intelligence

Finally, observe that

***∆*** *B* + ***Λ*** *[−]* *BB* [1] ***[Λ]*** *[BA]* ***[∆]*** *[A]* [ =] ***[ µ]*** *[B]* *[ −]* ***[Λ]*** *[−]* *BB* [1] ***[Λ]*** *[BA]* [(] ***[x]*** *[A]* *[ −]* ***[µ]*** *[A]* [) =] ***[ µ]*** *[B]* *[|]* *[A]* and using the third hint

***Λ*** *[−]* *BB* [1] [=] ***[ Σ]*** *[BB]* *[ −]* ***[Σ]*** *[BA]* ***[Σ]*** *[−]* *AA* [1] ***[Σ]*** *[AB]* [ =] ***[ Σ]*** *[B]* *[|]* *[A]* [.] using the second hint

Thus, ***x*** *B* *|* ***x*** *A* *∼N* ( ***µ*** *B* *|* *A*, ***Σ*** *B* *|* *A* ) .

**Solution to exercise 1.76 (Expectation and Variance of Gaussians).**
Note that if *Y* *∼N* ( 0, 1 ) is a (univariate) standard normal random
variable, then


1
**E** [ *Y* ] =
*√* 2 *π*


∞

*−* [1]
*−* ∞ *[y]* *[ ·]* [ exp] � 2


∞
� *−*



[1]

*dy* using the PDF of the standard normal
2 *[y]* [2] � distribution (1.6)



[1] ∞

*dy* +
2 *[y]* [2] � � 0


∞

*y* *·* exp *−* [1]
0 � 2



[1]

*dy*
2 *[y]* [2] � �


1

=
*√* 2 *π*


0
� � *−*


0

*−* [1]
*−* ∞ *[y]* *[ ·]* [ exp] � 2



[1] ∞

2 *[y]* [2] �� 0


0


1

=
*√* 2 *π*


*−* *−* [1]
exp
�� � 2


0

[1]

2 *[y]* [2] ��


*−* *−* [1]
+ exp
*−* ∞ � � 2


�


1
= ([ *−* 1 + 0 ] + [ 0 + 1 ]) = 0,
*√* 2 *π*


Var [ *Y* ] = **E** � *Y* [2] [�] *−* **E** ���� [ *Y* ] [2]
0


using the definition of variance (1.45)


1

=
*√* 2 *π*


∞

*−* [1]
*−* ∞ *[y]* [2] *[ ·]* [ exp] � 2


∞
� *−*



[1]

*dy* using the PDF of the standard normal
2 *[y]* [2] � distribution (1.6)



[1] ∞

*dy* +
2 *[y]* [2] � � 0


∞

*y* [2] *·* exp *−* [1]
0 � 2



[1]

*dy*
2 *[y]* [2] � �


1

=
*√* 2 *π*


0
� � *−*


0

*−* [1]
*−* ∞ *[y]* [2] *[ ·]* [ exp] � 2



[1]

*dy*
2 *[y]* [2] �


1

=
*√* 2 *π*


*−* *y* *·* exp *−* [1]
�� � 2


0

[1]

2 *[y]* [2] � �


0
+
*−* ∞ � *−*


0

*−* [1]
*−* ∞ [exp] � 2


integrating by parts



[1] ∞

2 *[y]* [2] �� 0


+ *−* *y* *·* exp *−* [1]
� � 2


∞

+
0 � 0


∞

*−* [1]
exp
0 � 2



[1]

*dy*
2 *[y]* [2] � �



[1] ∞

*dy* +
2 *[y]* [2] � � 0


∞

*−* [1]
exp
0 � 2



[1]

*dy*
2 *[y]* [2] � �


1

=
*√* 2 *π*


0
� � *−*


0

*−* [1]
*−* ∞ [exp] � 2


1

=
*√* 2 *π*


∞

*−* [1]
*−* ∞ [exp] � 2


∞
� *−*



[1] *dy* = 1. a PDF integrates to 1

2 *[y]* [2] �


Recall from eq. (1.122) that we can express **X** *∼N* ( ***µ***, ***Σ*** ) as

1 / 2
**X** = ***Σ*** **Y** + ***µ***

where **Y** *∼N* ( **0**, ***I*** ) . Using that **Y** is a vector of independent univariate
standard normal random variables, we conclude that **E** [ **Y** ] = **0** and
Var [ **Y** ] = ***I*** . We obtain


**E** [ **X** ] = **E** ***Σ*** 1 / 2 **Y** + ***µ*** = ***Σ*** 1 / 2 **E** [ **Y** ] + ***µ*** = ***µ***, and using linearity of expectation (1.24)
� � ����
**0**


Var [ **X** ] = Var ***Σ*** 1 / 2 **Y** + ***µ*** = ***Σ*** 1 / 2 Var [ **Y** ]
� � ����
***I***


***Σ*** 1 / 2 *[⊤]* = ***Σ*** . using eq. (1.47)

solutions 275


*Bayesian Linear Regression*

**Solution to exercise 2.2 (Closed-form linear regression).**

1. We begin by deriving the gradient and Hessian of the least squares
and ridge losses. For least squares,

***∇*** ***w*** *∥* ***y*** *−* ***Xw*** *∥* [2] 2 [=] ***[ ∇]*** ***[w]*** [ (] ***[w]*** *[⊤]* ***[X]*** *[⊤]* ***[Xw]*** *[ −]* [2] ***[w]*** *[⊤]* ***[X]*** *[⊤]* ***[y]*** [ +] *[ ∥]* ***[y]*** *[∥]* [2] 2 [)]

= 2 ( ***X*** *[⊤]* ***Xw*** *−* ***X*** *[⊤]* ***y*** ),

***H*** ***w*** *∥* ***y*** *−* ***Xw*** *∥* [2] 2 [=] [ 2] ***[X]*** *[⊤]* ***[X]*** [.]

Similarly, for ridge regression,

***∇*** ***w*** *∥* ***y*** *−* ***Xw*** *∥* [2] 2 [+] *[ λ]* *[ ∥]* ***[w]*** *[∥]* 2 [2] [=] [ 2] [(] ***[X]*** *[⊤]* ***[Xw]*** *[ −]* ***[X]*** *[⊤]* ***[y]*** [) +] *[ λ]* ***[w]*** [,]

***H*** ***w*** *∥* ***y*** *−* ***Xw*** *∥* [2] 2 [+] *[ λ]* *[ ∥]* ***[w]*** *[∥]* 2 [2] [=] [ 2] ***[X]*** *[⊤]* ***[X]*** [ +] *[ λ]* ***[I]*** [.]

From the assumption that the Hessians are positive definite, we
know that any minimizer is a unique globally optimal solution
(due to strict convexity), and that ***X*** *[⊤]* ***X*** and ***X*** *[⊤]* ***X*** + *λ* ***I*** are invert
ible.

Using the first-order optimality condition for convex functions, we
attain the solutions to least squares and ridge regression by setting
the respective gradient to **0** .
2. We choose ***w*** ˆ ls such that ***X*** ˆ ***w*** ls = ***Π*** ***X*** ***y*** . This implies that ***y*** *−*
***X*** ˆ ***w*** ls *⊥* ***Xw*** for all ***w*** *∈* **R** *[d]* . In other words, ( ***y*** *−* ***X*** ˆ ***w*** ls ) *[⊤]* ***Xw*** = 0
for all ***w***, which implies that ( ***y*** *−* ***X*** ˆ ***w*** ls ) *[⊤]* ***X*** = **0** . By simple algebraic manipulation it can be seen that this condition is equivalent
to the gradient condition of the previous exercise.

**Solution to exercise 2.4 (Variance of least squares around training**
**data).** In the two-dimensional setting, i.e., data is of the form ***x*** *i* =

[ 1 *x* *i* ] *[⊤]* ( *x* *i* *∈* **R** ), we have


�


***X*** *[⊤]* ***X*** =


*n* ∑ *x* *i*
�∑ *x* *i* ∑ *x* *i* [2]


.


Thus,

*n*
Var [ ˆ ***w*** ls ] = *σ* *n* [2] [(] ***[X]*** *[⊤]* ***[X]*** [)] *[−]* [1] [ =] *[σ]* [2]
*Z*

where *Z* = . *n* ( ∑ *x* *i* 2 [)] *[ −]* [(] [∑] *[x]* *[i]* [)] [2] [.]


∑ *x* *i* [2] *−* ∑ *x* *i*
*−* ∑ *x* *i* *n*
�


�


using eq. (2.8)


Therefore, the predictive variance at a point [ 1 *x* *[⋆]* ] *[⊤]* is


�


1 *x* *[⋆]* [�] Var [ ˆ ***w*** ls ]
�


1

*x* *[⋆]*
�


= *[σ]* *n* [2] *i* *[−]* [2] *[x]* *[i]* *[x]* *[⋆]* [+ (] *[x]* *[⋆]* [)] [2] [ =] *[σ]* *n* [2]
###### Z [∑] [x] [2] Z [∑] [(] [x] [i] [ −] [x] [⋆] [)] [2] [.]


Thus, the predictive variance is minimized for *x* *[⋆]* = *n* [1] [∑] *[x]* *[i]* [.]

276 probabilistic artificial intelligence

**Solution to exercise 2.9 (Aleatoric and epistemic uncertainty).** The
law of total variance (2.6) yields the following decomposition of the
predictive variance,

Var *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ] = **E** ***w*** �Var *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***w*** ] �� ***x*** 1: *n*, *y* 1: *n* �

+ Var ***w*** � **E** *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***w*** ] �� ***x*** 1: *n*, *y* 1: *n* �

wherein the first term corresponds to the aleatoric uncertainty and the
second term corresponds to the epistemic uncertainty.

The aleatoric uncertainty is given by

**E** ***w*** �Var *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***w*** ] �� ***x*** 1: *n*, *y* 1: *n* � = **E** ***w*** � *σ* *n* [2] ��� ***x*** 1: *n*, *y* 1: *n* � = *σ* *n* [2] using the definition of *σ* *n* [2] [. (][2][.][10][)]

For the epistemic uncertainty,

Var ***w*** � **E** *y* *⋆* [ *y* *[⋆]* *|* ***x*** *[⋆]*, ***w*** ] �� ***x*** 1: *n*, *y* 1: *n* � = Var ***w*** � ***w*** *[⊤]* ***x*** *[⋆]* [��] � ***x*** 1: *n*, *y* 1: *n* � using that *y* *[⋆]* = ***w*** *[⊤]* ***x*** *[⋆]* + *ϵ* where *ϵ* is
zero-mean noise

= ***x*** *[⋆][⊤]* Var ***w*** [ ***w*** *|* ***x*** 1: *n*, *y* 1: *n* ] ***x*** *[⋆]* using eq. (1.47)

= ***x*** *[⋆][⊤]* ***Σx*** *[⋆]* using eq. (2.16)

where ***Σ*** is the posterior covariance matrix.

**Solution to exercise 2.11 (Bayesian linear regression).**

1. Recall from example 2.6 that the MLE and least squares estimate
coincide if the noise is zero-mean Gaussian. We therefore have,


�


***w*** ˆ MLE = ˆ ***w*** ls = ( ***X*** *[⊤]* ***X*** ) *[−]* [1] ***X*** *[⊤]* ***y*** =


0.63

1.83
�


.


2. Recall from eq. (2.16) that ***w*** *|* ***X***, ***y*** *∼N* ( ***µ***, ***Σ*** ) with

*−* 1
***Σ*** = � *σ* *n* *[−]* [2] ***[X]*** *[⊤]* ***[X]*** [ +] *[ σ]* *p* *[−]* [2] ***[I]*** � and

***µ*** = *σ* *n* *[−]* [2] ***[Σ][X]*** *[⊤]* ***[y]*** [.]

Inserting *σ* *n* [2] [=] [ 0.1 and] *[ σ]* *p* [2] [=] [ 0.05 yields,]


***Σ*** =


0.019 *−* 0.014

*−* 0.014 0.019
�


.


Then,


�


�

.


***w*** ˆ MAP = ***µ*** =


0.91

1.31
�


3. Recall from eq. (2.22) that *y* *[⋆]* *|* ***X***, ***y***, ***x*** *[⋆]* *∼N* ( *µ* *[⋆]* *y* [,] *[ σ]* *y* [2] *[⋆]* [)] [ with]

*µ* *[⋆]* *y* [=] ***[ µ]*** *[⊤]* ***[x]*** *[⋆]* and *σ* *y* [2] *[⋆]* [=] ***[ x]*** *[⋆][⊤]* ***[Σ][x]*** *[⋆]* [+] *[ σ]* *n* [2] [.]

Inserting ***x*** *[⋆]* = [ 3 3 ] *[⊤]*, and *σ* *n* [2] [,] ***[ µ]*** [ and] ***[ Σ]*** [ from above yields,]

*µ* *[⋆]* *y* [=] [ 6.66] and *σ* *y* [2] *[⋆]* [=] [ 0.186.]

4. One would have to let *σ* *p* [2] *[→]* [∞][.]

solutions 277


**Solution to exercise 2.12 (Online Bayesian linear regression).** We
denote by ***X*** *t* the design matrix and by ***y*** *t* the vector of observations
including the first *t* data points.

1. Note that

*t* *t*
###### X t [⊤] [X] [t] [=] ∑ x i x i [⊤] and X t [⊤] [y] [t] [=] ∑ y i x i .

*i* = 1 *i* = 1

This means that after observing the ( *t* + 1 ) -st data point, we have

that

***X*** *t* *[⊤]* + 1 ***[X]*** *[t]* [+] [1] [=] ***[ X]*** *t* *[⊤]* ***[X]*** *[t]* [+] ***[ x]*** *t* + 1 ***[x]*** *[⊤]* *t* + 1 and

***X*** *t* *[⊤]* + 1 ***[y]*** *[t]* [+] [1] [=] ***[ X]*** *t* *[⊤]* ***[y]*** *[t]* [+] *[ y]* *t* + 1 ***[x]*** *t* + 1 [.]

Hence, by just keeping ***X*** *t* *[⊤]* ***[X]*** *[t]* [ (which is a] *[ d]* *[ ×]* *[ d]* [ matrix) and] ***[ X]*** *t* *[⊤]* ***[y]*** *[t]*
(which is a vector in **R** *[d]* ) in memory, and updating them as above,
we do not need to keep the whole data in memory.
2. One has to compute ( *σ* *n* *[−]* [2] ***[X]*** *t* *[⊤]* ***[X]*** *[t]* [ +] *[ σ]* *p* *[−]* [2] ***[I]*** [)] *[−]* [1] [ for finding] ***[ µ]*** [ and] ***[ Σ]*** [ in]
every round. We can write

( *σ* *n* *[−]* [2] ***[X]*** *t* *[⊤]* + 1 ***[X]*** *[t]* [+] [1] [+] *[ σ]* *p* *[−]* [2] ***[I]*** [)] *[−]* [1] [ =] *[ σ]* *n* [2] [(] ***[X]*** *t* *[⊤]* + 1 ***[X]*** *[t]* [+] [1] [+] *[ σ]* *n* [2] *[σ]* *p* *[−]* [2] ***[I]*** [)] *[−]* [1]


= *σ* *n* [2] [(] ***[X]*** *t* *[⊤]* ***[X]*** *[t]* [+] *[ σ]* *n* [2] *[σ]* *p* *[−]* [2] ***[I]***
� �� �
***A*** *t*


+ ***x*** *t* + 1 ***x*** *[⊤]* *t* + 1 [)] *[−]* [1]


where ***A*** *t* *∈* **R** *[d]* *[×]* *[d]* . Using the Woodbury matrix identity (2.28) and
that we know the inverse of ***A*** *t* (from the previous iteration), the
computation of the inverse of ( ***A*** *t* + ***x*** *t* + 1 ***x*** *[⊤]* *t* + 1 [)] [ is of] *[ O]* � *d* [2] [�], which
is much better than computing the inverse of ( ***A*** *t* + ***x*** *t* + 1 ***x*** *[⊤]* *t* + 1 [)] [ from]
scratch.

*Kalman Filters*

**Solution to exercise 3.6 (Kalman update).** Recall from eq. (3.10) that

*p* ( *x* *t* + 1 *|* *y* 1: *t* + 1 ) = [1] (A.2)

*Z* *[p]* [(] *[x]* *[t]* [+] [1] *[ |]* *[ y]* [1:] *[t]* [)] *[p]* [(] *[y]* *[t]* [+] [1] *[ |]* *[ x]* *[t]* [+] [1] [)] [.]

Using the sensor model (3.14b),


.

�


*p* ( *y* *t* + 1 *|* *x* *t* + 1 ) = [1]

*Z* *[′]* [ exp]


�


*−* [1]



[1] ( *y* *t* + 1 *−* *x* *t* + 1 ) [2]

2 *σ* [2]


*σ* [2]
*y*


It remains to compute the predictive distribution,

*p* ( *x* *t* + 1 *|* *y* 1: *t* ) = *p* ( *x* *t* + 1 *|* *x* *t* ) *p* ( *x* *t* *|* *y* 1: *t* ) *dx* *t* using eq. (3.11)
�


*σ* [2]
*t*


*dx* *t* using the motion model (3.14a) and
�� previous update


( *x* *t* + 1 *−* *x* *t* ) 2
� *σ* *x* [2]


1 *−* *x* *t* ) 2 + [(] *[x]* *[t]* *[ −]* *[µ]* *[t]* [)] [2]

*σ* *x* [2] *σ* [2]


= [1]

*Z* *[′′]*


*−* [1]
exp
� � 2

278 probabilistic artificial intelligence


2
*σ* *t* [(] *[x]* *[t]* [+] [1] *[−]* *[x]* *[t]* [)] [2] [ +] *[ σ]* *x* [2] [(] *[x]* *[t]* *[−]* *[µ]* *[t]* [)] [2]

*dx* *t* .

� *σ* *t* [2] *[σ]* *x* [2] ��


= [1]

*Z* *[′′]*


*−* [1]
exp
� � 2


The exponent is the sum of two expressions that are quadratic in *x* *t* .
*Completing the square* allows rewriting any quadratic *ax* [2] + *bx* + *c* as
the sum of a squared term *a* ( *x* + 2 *[b]* *a* [)] [2] [ and a residual term] *[ c]* *[ −]* 4 *[b]* [2] *a* [that]



*[b]*

2 *a* [)] [2] [ and a residual term] *[ c]* *[ −]* 4 *[b]* [2] *a*


the sum of a squared term *a* ( *x* + 2 *a* [)] [ and a residual term] *[ c]* *[ −]* 4 *a* [that]

is independent of *x* . In this case, we have *a* = ( *σ* *t* [2] [+] *[ σ]* *x* [2] [)] [/] [(] *[σ]* *t* [2] *[σ]* *x* [2] [)] [,]
*b* = *−* 2 ( *σ* *t* [2] *[x]* *[t]* [+] [1] [ +] *[ σ]* *x* [2] *[µ]* *[t]* [)] [/] [(] *[σ]* *t* [2] *[σ]* *x* [2] [)] [, and] *[ c]* [ = (] *[σ]* *t* [2] *[x]* *t* [2] + 1 [+] *[ σ]* *x* [2] *[µ]* [2] *t* [)] [/] [(] *[σ]* *t* [2] *[σ]* *x* [2] [)] [. The]
residual term can be taken outside the integral, giving


exp
�� � �



[1] *−* [1]

*Z* *[′′]* [ exp] � 2


� 2 [�]


= [1]


2


*x* *t* + *[b]*
� 2 *a*


*c* *−* *[b]* [2]
� 4 *a*


*−* *[a]*

2


*dx* *t* .


The integral is simply the integral of a Gaussian over its entire support,
and thus evaluates to 1. We are therefore left with only the residual
term from the quadratic. Plugging back in the expressions for *a*, *b*, and
*c* and simplifying, we obtain


.
�



[1] *−* [1]

*Z* *[′′]* [ exp] � 2


*σ* *t* [2] [+] *[ σ]* *x* [2]


= [1]



[1] ( *x* *t* + 1 *−* *µ* *t* ) [2]

2 *σ* [2] *[ σ]* [2]


That is, *X* *t* + 1 *|* *y* 1: *t* *∼N* ( *µ* *t*, *σ* *t* [2] [+] *[ σ]* *x* [2] [)] [.]

Plugging our results back into eq. (A.2), we obtain


�


.

��


1
*p* ( *x* *t* + 1 *|* *y* 1: *t* + 1 ) =
*Z* *[′′′]* [ exp]


*−* [1]

2


( *x* *t* + 1 *−* *µ* *t* ) [2]
� *σ* *t* [2] [+] *[ σ]* *x* [2]


*t* + 1 *−* *µ* *t* ) [2] + [(] *[y]* *[t]* [+] [1] *[−]* *[x]* *[t]* [+] [1] [)] [2]

*σ* *t* [2] [+] *[ σ]* *x* [2] *σ* *y* [2]


*σ* [2]
*y*


Completing the square analogously to our derivation of the predictive
distribution yields,







 [.]





� 2


1

=
*Z* *[′′′]* [ exp]


2




 2
 *[−]* [1]


( *σ* *t* [2] [+] *[σ]* *x* [2] [)] *[y]* *[t]* [+] [1] [+] *[σ]* *y* [2] *[µ]* *[t]*
� *x* *t* + 1 *−* *σ* *t* [2] [+] *[σ]* *[x]* [2] [+] *[σ]* *[y]* [2]

( *σ* *t* [2] [+] *[σ]* *[x]* [2] [)] *[σ]* *[y]* [2]
*σ* *t* [2] [+] *[σ]* *[x]* [2] [+] *[σ]* *[y]* [2]


Hence, *X* *t* + 1 *|* *y* *t* + 1 *∼N* ( *µ* *t* + 1, *σ* *t* [2] + 1 [)] [ as defined in eq. (][3][.][15][).]

**Solution to exercise 3.7 (Parameter estimation using Kalman filters).**

1. The given parameter estimation problem can be formulated as a
Kalman filter in the following way:

*x* *t* = *π* *∀* *t*,

*y* *t* = *x* *t* + *η* *t* *η* *t* *∼N* ( 0, *σ* *y* [2] [)] [.]

Thus, in terms of Kalman filters, this yields *f* = *h* = 1, *ϵ* *t* = *σ* *x* [2] [=] [ 0.]
Using eq. (3.16), the *Kalman gain* is given by

*k* *t* + 1 = *σ* *t* [2],
*σ* *t* [2] [+] *[ σ]* *y* [2]

solutions 279


whereas the *variance of the estimation error σ* *t* [2] [satisfies]

*σ* *t* [2] *[σ]* *y* [2]
*σ* *t* [2] + 1 [=] *[ σ]* *y* [2] *[k]* *t* + 1 [=] . using eq. (3.19)
*σ* *t* [2] [+] *[ σ]* *y* [2]

To get the closed form, observe that


1
= [1]
*σ* [2] *σ*
*t* + 1 *t*



[1] + [1]

*σ* [2] *σ*
*t*



[2] = *· · ·* = [1]

*σ* [2] *σ*
*y*


1

[1] = + [2]

*σ* *y* [2] *σ* *t* [2] *−* 1 *σ*



[1] + *[t]* [ +] [ 1]

*σ* [2] *σ* [2]
0 *y*


,
*σ* [2]
*y*


yielding,

*σ* 0 [2] *[σ]* *y* [2] *σ* 0 [2]
*σ* *t* [2] + 1 [=] ( *t* + 1 ) *σ* 0 [2] [+] *[ σ]* *[y]* [2] and *k* *t* + 1 = ( *t* + 1 ) *σ* 0 [2] [+] *[ σ]* *[y]* [2] .

2. When *t* *→* ∞, we get *k* *t* + 1 *→* 0 and *σ* *t* [2] + 1 *[→]* [0, giving]

*µ* *t* + 1 = *µ* *t* + *k* *t* + 1 ( *y* *t* + 1 *−* *µ* *t* ) = *µ* *t*, using eq. (3.18)

thus resulting in a stationary sequence.
1
3. Observe that *σ* 0 [2] *[→]* [∞] [implies] *[ k]* *[t]* [+] [1] [ =] *t* + 1 [. Therefore,]

1
*µ* *t* + 1 = *µ* *t* + using eq. (3.18)
*t* + 1 [(] *[y]* *[t]* [+] [1] *[ −]* *[µ]* *[t]* [)]

*t*
= *[y]* *[t]* [+] [1]
*t* + 1 *[µ]* *[t]* [ +] *t* + 1


*t*

=
*t* + 1

= *[t]* *[ −]* [1]


*t* *−* 1
� *t*


*µ* *t* *−* 1 + *[y]* *[t]*
*t* *t*


*t*


+ *[y]* *[t]* [+] [1] using eq. (3.18)
� *t* + 1



*[t]* *[ −]* [1] *[y]* *[t]* [ +] *[y]* *[t]* [+] [1]

*t* + 1 *[µ]* *[t]* *[−]* [1] [ +] *t* + 1


*t* + 1 *t* + 1

...



[+] *[ · · ·]* [ +] *[y]* *[t]* [+] [1]
= *[y]* [1] *t* + 1,


which is simply the *sample mean* .

*Gaussian Processes*

**Solution to exercise 4.4 (Feature space of Gaussian kernel).**

1. We have for any *j* *∈* **N** 0 and *x*, *y* *∈* **R**,



[+] 2 *[y]* [2] ( *x* *y* ) *[j]*


*j* ! .


1

�



[2] 1

2 *x* *[j]* *·*
�



[2]

2 *y* *[j]* = *e* *[−]* *[x]* [2] [+] 2 *[y]* [2]


2

*j* ! *[e]* *[−]* *[x]* [2]


2

*j* ! *[e]* *[−]* *[y]* [2]


Summing over all *j*, we obtain

∞
###### ϕ ( x ) [⊤] ϕ ( y ) = ∑ ϕ j ( x ) ϕ j ( y )

*j* = 0

280 probabilistic artificial intelligence


∞

= *e* *[−]* *[x]* [2] [+] 2 *[y]* [2]
###### ∑

*j* = 0


( *x* *y* ) *[j]*

*j* !


= *e* *[−]* *[x]* [2] [+] 2 *[y]* [2]


= *e* *[−]* 2 *e* *[xy]* using the Taylor series expansion for the

exponential function
= *e* *[−]* [(] *[x]* *[−]* 2 *[y]* [)] [2]


2


= *k* ( *x*, *y* ) .

2. As we have seen in section 2.6, the effective dimension is *n* . The

crucial difference of kernelized regression (e.g., Gaussian processes)
to linear regression is that the effective dimension *grows* with the
sample size, whereas it is fixed for linear regression. Models where
the effective dimension may depend on the sample size are called
*non-parametric* models, and models where the effective dimension
is fixed are called *parametric* models.

**Solution to exercise 4.5 (A Kalman filter as a Gaussian process).**
First we look at the mean,

*µ* ( *t* ) = **E** [ *X* *t* ] = **E** [ *X* *t* *−* 1 + *ϵ* *t* *−* 1 ] = **E** [ *X* *t* *−* 1 ] = *µ* ( *t* *−* 1 ) .

Knowing that *µ* ( 0 ) = 0, we can derive that *µ* ( *t* ) = 0 ( *∀* *t* ) .

Now we look at the variance of *X* *t*,


*t* *−* 1
###### X 0 + ∑ ϵ τ
*τ* = 0 �


Var [ *X* *t* ] = Var


�


= *σ* 0 [2] [+] *[ t][σ]* *x* [2] [.] using that the noise is independent


Finally, we look at the distribution of [ *f* ( *t* ) *f* ( *t* *[′]* )] *[⊤]* for arbitrary *t* *≤* *t* *[′]* .


*X* *t*
� *X* *t* *′* �

Therefore, we get that


�


0

*ϵ* *τ*
� �


=


*X* *t*

*X* *t*
�


*t* *[′]* *−* 1
###### + ∑

*τ* = *t*


.


*X* *t*
� *X* *t* *′*


�


*∼N*

= *N*


�

�


+ ( *t* *[′]* *−* *t* )

�


0 0
�0 *σ* *x* [2]


��


0

0
��

0

0
��


,


, Var [ *X* *t* ]


1 1

1 1
�


��


Var [ *X* *t* ] Var [ *X* *t* ]
�Var [ *X* *t* ] Var [ *X* *t* ] + ( *t* *[′]* *−* *t* ) *σ* *x* [2]


.


We take the kernel *k* KF ( *t*, *t* *[′]* ) to be the covariance between *f* ( *t* ) and
*f* ( *t* *[′]* ), which is Var [ *X* *t* ] = *σ* 0 [2] [+] *[ σ]* *x* [2] *[t]* [. Notice, however, that we assumed]
*t* *≤* *t* *[′]* . Thus, overall, the kernel is described by

*k* KF ( *t*, *t* *[′]* ) = *σ* 0 [2] [+] *[ σ]* *x* [2] [min] *[{]* *[t]* [,] *[ t]* *[′]* *[}]* [.]

solutions 281


**Solution to exercise 4.9 (RKHS norm).**

1. As *f* *∈H* *k* ( *X* ) we can express *f* ( ***x*** ) for some *β* *i* *∈* **R** and ***x*** *i* *∈X* as

*n*
###### f ( x ) = ∑ β i k ( x, x i )

*i* = 1

*n*
###### = ∑ β i ⟨ k ( x i, · ), k ( x, · ) ⟩ k

*i* = 1


�


=


*n*
###### ∑ β i k ( x i, · ), k ( x, · )
� *i* = 1


= *k*

= *⟨* *f* ( *·* ), *k* ( ***x***, *·* ) *⟩* *k* .


2. By applying Cauchy-Schwarz,

*|* *f* ( ***x*** ) *−* *f* ( ***y*** ) *|* = *| ⟨* *f*, *k* ( ***x***, *·* ) *−* *k* ( ***y***, *·* ) *⟩* *k* *|*

*≤∥* *f* *∥* *k* *∥* *k* ( ***x***, *·* ) *−* *k* ( ***y***, *·* ) *∥* *k*

**Solution to exercise 4.11 (MAP estimate of Gaussian processes).**

1. By the representer theorem, *f* [ˆ] ( ***x*** ) = ˆ ***α*** *[⊤]* ***k*** ***x***, *A* . In particular, we have
***f*** = ***Kα*** ˆ and therefore

1
*−* log *p* ( *y* 1: *n* *|* ***x*** 1: *n*, *f* ) = 2 *σ* *n* [2] *∥* ***y*** *−* ***f*** *∥* [2] 2 [+] [ const]

= 21 *σ* *n* [2] *∥* ***y*** *−* ***Kα*** ˆ *∥* [2] 2 [+] [ const.]

The regularization term simplifies to

*∥* *f* *∥* [2] *k* [=] *[ ⟨]* *[f]* [,] *[ f]* *[ ⟩]* *k* [=] [ ˆ] ***[α]*** *[⊤]* ***[K][α]*** [ˆ] [ =] *[ ∥]* ***[α]*** [ˆ] *[∥]* [2] ***K*** [.]

Combining, we have that

ˆ 1
***α*** = arg min *∥* ***y*** *−* ***Kα*** *∥* [2] 2 [+] [1] ***K***
***α*** *∈* **R** *[n]* 2 *σ* *n* [2] 2 *[∥]* ***[α]*** *[∥]* [2]

as desired. It follows by multiplying through with 2 *σ* *n* [2] [that] *[ λ]* [ =] *[ σ]* *n* [2] [.]
2. Expanding the objective determined in (1), we are looking for the

minimizer of

***α*** *[⊤]* ( *σ* *n* [2] ***[K]*** [ +] ***[ K]*** [2] [)] ***[α]*** *[ −]* [2] ***[y]*** *[⊤]* ***[K][α]*** [ +] ***[ y]*** *[⊤]* ***[y]*** [.]

Differentiating with respect to the coefficients ***α***, we obtain the minimizer ˆ ***α*** = ( ***K*** + *σ* *n* [2] ***[I]*** [)] *[−]* [1] ***[y]*** [. Thus, the prediction at a point] ***[ x]*** *[⋆]* [is]
***k*** *[⊤]* ***x*** *[⋆]*, *A* [(] ***[K]*** [ +] *[ σ]* *n* [2] ***[I]*** [)] *[−]* [1] ***[y]*** [ which coincides with the MAP estimate.]

**Solution to exercise 4.13 (Gradient of the marginal likelihood).**

1. Applying the two hints eqs. (4.42) and (4.43) to eq. (4.40) yields,


*∂* ***K*** ***y***, ***θ***

[1]

2 ***[y]*** *[⊤]* ***[K]*** ***y*** *[−]*, ***θ*** [1] *∂θ*


2 [tr]


.

�


*∂* log *p* ( ***y*** *|* ***X***, ***θ*** ) = [1]
*∂θ* 2
*j*


***K*** ***y***, ***θ*** ***K*** *[−]* [1]

*∂θ* ***y***, ***θ*** ***[y]*** *[ −]* [1] 2
*j*


�


***K*** *[−]* [1] *∂* ***K*** ***y***, ***θ***
***y***, ***θ*** *∂θ*

*j*

282 probabilistic artificial intelligence

We can simplify to


.

�


using that ***y*** *[⊤]* ***K*** ***y*** *[−]*, ***θ*** [1] *∂* ***K*** *∂θ* ***y*** *j*, ***θ*** ***[K]*** ***y*** *[−]*, ***θ*** [1] ***[y]*** [ is a scalar]

using the cyclic property and linearity
of the trace

using that ***K*** ***y*** *[−]*, ***θ*** [1] [is symmetric]


�


*−* [1]

2 [tr]


�


= [1]

2 [tr]


= [1]


***y*** *[⊤]* ***K*** ***y*** *[−]*, ***θ*** [1] *∂* ***K*** *∂θ* ***y***, ***θ*** ***K*** ***y*** *[−]*, ***θ*** [1] ***[y]***

*j*


***y*** *[⊤]* ***K*** ***y*** *[−]*, ***θ*** [1] *∂* ***K*** ***y***, ***θ***


�


***K*** *[−]* [1] *∂* ***K*** ***y***, ***θ***
***y***, ***θ*** *∂θ*

*j*


*∂θ*
*j*


= [1]

2 [tr]


= [1]


***K*** *[−]* [1] *∂* ***K*** ***y***, ***θ***
***y***, ***θ*** ***[yy]*** *[⊤]* ***[K]*** ***y*** *[−]*, ***θ*** [1]


***K*** ***y***, ***θ*** *−* ***K*** *[−]* [1] *∂* ***K*** ***y***, ***θ***

*∂θ* ***y***, ***θ*** *∂θ*
*j* *j*


�


*∂θ*
*j*


�


= [1]

2 [tr]


= [1]


***K*** *[−]* [1]
***y***, ***θ*** ***[y]*** [(] ***[K]*** ***y*** *[−]*, ***θ*** [1] ***[y]*** [)] *[⊤]* *[∂]* ***[K]*** ***[y]*** [,] ***[θ]***



***[K]*** ***[y]*** [,] ***[θ]*** *−* ***K*** *[−]* [1] *∂* ***K*** ***y***, ***θ***

*∂θ* ***y***, ***θ*** *∂θ*
*j* *j*


= [1]

2 [tr]


= [1]


�

�

�

�


( *αα* *[⊤]* *−* ***K*** ***y*** *[−]*, ***θ*** [1] [)] *[∂]* ***[K]*** *∂θ* ***[y]*** [,] ***[θ]***

*j*


( *αα* *[⊤]* *−* ***K*** ***y*** *[−]*, ***θ*** [1] [)] *[∂]* ***[K]*** ***[y]*** [,] ***[θ]***


.


2. We denote by ***K*** [˜] the covariance matrix of ***y*** for the covariance function *k* [˜], so ***K*** ***y***, ***θ*** = *θ* 0 ***K*** [˜] . Then,

*∂* ˜ ˜
log *p* ( ***y*** *|* ***X***, ***θ*** ) = [1] ( *θ* 0 *[−]* [2] ***K*** *[−]* [1] ***y*** ( ˜ ***K*** *[−]* [1] ***y*** ) *[⊤]* *−* *θ* 0 *[−]* [1] ***K*** *[−]* [1] ) ˜ ***K*** . using eq. (4.41)
*∂θ* 0 2 [tr] � �

Simplifying the terms and using linearity of the trace, we obtain

that

*∂* log *p* ( ***y*** *|* ***X***, ***θ*** ) = 0 *⇐⇒* *θ* 0 = [1] ***yy*** *[⊤]* ***K*** [˜] *[−]* [1] [�] .
*∂θ* 0 *n* [tr] �

If we define ***Λ*** [˜] = . ˜ ***K*** *−* 1 as the precision matrix associated to ***y*** for
the covariance function *k* [˜], we can express *θ* 0 *[⋆]* [in closed form as]


*n*
###### ∑

*i* = 1


*n*
###### ∑ Λ ˜ ( i, j ) y i y j . (A.3)

*j* = 1


*θ* 0 *[⋆]* [=] [1]

*n*


*θ* 0 *[⋆]* [=] [1]


3. We immediately see from eq. (A.3) that *θ* 0 *[⋆]* [scales by] *[ s]* [2] [ if] ***[ y]*** [ is scaled]
by *s* .

**Solution to exercise 4.19 (Uniform convergence of Fourier features).**


2, *√*


1. We have that *s* ( *·* ) *∈* [ *−√*


2 ], and hence, *s* ( ***∆*** *i* ) is *√*


. We have that *s* ( *·* ) *∈* [ *−√* 2, *√* 2 ], and hence, *s* ( ***∆*** *i* ) is *√* 2-sub-Gaussian. [1] 1 see example 1.48

It then follows from Hoeffding’s inequality (1.87) that


**P** ( *|* *f* ( ***∆*** *i* ) *| ≥* *ϵ* ) *≤* 2 exp *−* *[m][ϵ]* [2]
� 4


.
�


2. We can apply Markov’s inequality (1.81) to obtain


� 2 [�]


**P** *∥* ***∇*** *f* ( ***∆*** *[⋆]* ) *∥* 2 *≥* *[ϵ]*
� 2 *r*


**P** *∥* ***∇*** *f* ( ***∆*** *[⋆]* ) *∥* 2 *≥* *[ϵ]*
� 2


*ϵ*
= **P** *∥* ***∇*** *f* ( ***∆*** *[⋆]* ) *∥* [2] 2 *[≥]*
� � � 2 *r*


2

.
�


*≤* 2 *r* **E** *∥* ***∇*** *f* ( ***∆*** *⋆* ) *∥* 2
� *ϵ*

It remains to bound the expectation. We have


*≤* 2 *r* **E** *∥* ***∇*** *f* ( ***∆*** *⋆* ) *∥* 2
� *ϵ*


**E** *∥* ***∇*** *f* ( ***∆*** *[⋆]* ) *∥* [2] 2 [=] **[ E]** *[ ∥]* ***[∇]*** *[s]* [(] ***[∆]*** *[⋆]* [)] *[ −]* ***[∇]*** *[k]* [(] ***[∆]*** *[⋆]* [)] *[∥]* 2 [2]

solutions 283

= **E** *∥* ***∇*** *s* ( ***∆*** *[⋆]* ) *∥* [2] 2 *[−]* [2] ***[∇]*** *[k]* [(] ***[∆]*** *[⋆]* [)] *[⊤]* **[E]** ***[∇]*** *[s]* [(] ***[∆]*** *[⋆]* [) +] **[ E]** *[ ∥]* ***[∇]*** *[k]* [(] ***[∆]*** *[⋆]* [)] *[∥]* [2] 2 [.] using linearity of expectation (1.24)

Note that **E** ***∇*** *s* ( ***∆*** ) = ***∇*** *k* ( ***∆*** ) using eq. (1.28) and using that *s* ( ***∆*** ) is
an unbiased estimator of *k* ( ***∆*** ) . Therefore,

= **E** *∥* ***∇*** *s* ( ***∆*** *[⋆]* ) *∥* [2] 2 *[−]* **[E]** *[ ∥]* ***[∇]*** *[k]* [(] ***[∆]*** *[⋆]* [)] *[∥]* [2] 2
*≤* **E** *∥* ***∇*** *s* ( ***∆*** *[⋆]* ) *∥* [2] 2
*≤* **E** ***ω*** *∼* *p* *∥* ***ω*** *∥* [2] 2 [=] *[ σ]* *p* [2] [.] using that *s* is the cos of a linear
function in ***ω***

3. Using a union bound (1.1) and then the result of (1),


*T*
�
� *i* = 1


�


**P**


� *|* *f* ( ***∆*** *i* ) *| ≥* *[ϵ]*

2

*i* = 1


2


*T*
###### ≤ i ∑ = 1 P � | f ( ∆ i ) | ≥ 2 [ϵ]


*≤* 2 *T* exp *−* *[m][ϵ]* [2] .
� � 16 �


4. First, note that by contraposition,


sup *|* *f* ( ***∆*** ) *| ≥* *ϵ* = *⇒∃* *i* . *|* *f* ( ***∆*** *i* ) *| ≥* *[ϵ]*
***∆*** *∈M* ∆



*[ϵ]*

2 [or] *[ ∥]* ***[∇]*** *[f]* [ (] ***[∆]*** *[⋆]* [)] *[∥]* [2] *[ ≥]* 2 *[ϵ]* *r*


2 *r* [,]


and therefore,


�


*≤* **P**

*≤* **P**


*T*
�
� *i* =

*T*
�
� *i* =


�


**P**


�


sup *|* *f* ( ***∆*** ) *| ≥* *ϵ*
***∆*** *∈M* ∆


� *|* *f* ( ***∆*** *i* ) *| ≥* *[ϵ]*

2

*i* = 1


� *|* *f* ( ***∆*** *i* ) *| ≥* *[ϵ]*

2

*i* = 1



*[ϵ]*

2 *[∪∥]* ***[∇]*** *[f]* [ (] ***[∆]*** *[⋆]* [)] *[∥]* [2] *[ ≥]* 2 *[ϵ]* *r*


2


2 *r*


�


+ **P** *∥* ***∇*** *f* ( ***∆*** *[⋆]* ) *∥* 2 *≥* *[ϵ]*
� 2 *r*


+ **P** *∥* ***∇*** *f* ( ***∆*** *[⋆]* ) *∥* 2 *≥* *[ϵ]*
� 2


using a union bound (1.1)
�


*≤* 2 *T* exp *−* *[m][ϵ]* [2]
� 2 [4]


2 *rσ* *p* 2
+ using the results from (2) and (3)
� � *ϵ* �


*≤* *αr* *[−]* *[d]* + *βr* [2] using *T* *≤* ( 4 diam ( *M* ) / *r* ) *[d]*


with *α* = 2 ( 4 diam ( *M* )) *[d]* exp� *−* *[m]* 2 *[ϵ]* [4][2] � and *β* = � 2 *σϵ* *p* � 2 . Using the

hint, we obtain


with *α* = 2 ( 4 diam ( *M* )) *[d]* exp� *−* *[m]* 2 *[ϵ]* [4][2]



*[ϵ]* [2] 2 *σ* *p*

2 [4] � and *β* = � *ϵ*


*d* 2
= 2 *β* *d* + 2 *α* *d* + 2


�

using *[σ]* *[p]* [diam] *ϵ* [(] *[M]* [)] *≥* 1


= 22




2 [3] *σ* *p* dia g ( *M* )

*ϵ*

�


� 2 *d* [] 


� 2 *d* [] 


1
*d* + 2
*mϵ* [2]
exp *−*
� 2 [3] ( *d* + 2 )


1
*d* + 2
*mϵ* [2]
exp *−*
� 2 [3] ( *d* +


2
*mϵ* [2]
exp *−*
� � 2 [3] ( *d* + 2 )


2
*mϵ* [2]
exp *−*
� � 2 [3] ( *d* +


�


*≤* 2 [8] *σ* *p* dia g ( *M* )
� *ϵ*


5. We have


*≤* 2 [8] *σ* *p* dia g ( *M* )
� *ϵ*

*σ* *p* [2] [=] **[ E]** ***[ω]*** *[∼]* *[p]* � ***ω*** *[⊤]* ***ω*** �


= ***ω*** *[⊤]* ***ω*** *·* *p* ( ***ω*** ) *d* ***ω***
�

= ***ω*** *[⊤]* ***ω*** *·* *e* *[i]* ***[ω]*** *[⊤]* **[0]** *p* ( ***ω*** ) *d* ***ω*** .
�

284 probabilistic artificial intelligence


Now observe that *[∂]* [2]

*∂* ∆ [2]
*j*


� *e* *[i]* ***[ω]*** *[⊤]* ***[∆]*** *p* ( ***ω*** ) *d* ***ω*** = *−* [�] *ω* [2] *j* *[e]* *[i]* ***[ω]*** *[⊤]* ***[∆]*** *[p]* [(] ***[ω]*** [)] *[ d]* ***[ω]*** [. Thus,]


= *−* tr ***H*** ***∆***
�


*p* ( ***ω*** ) *e* *[i]* ***[ω]*** *[⊤]* ***[∆]*** *d* ***ω***
� ���� ***∆*** = **0** �


= *−* tr ( ***H*** ***∆*** *k* ( **0** )) . using that *p* is the Fourier transform of
*k* (4.51)

Finally, we have for the Gaussian kernel that


*∂* [2]

exp
*∂* ∆ [2]
*j* �


*−* ***[∆]*** *[⊤]* ***[∆]*** = *−* [1]

2 *ℓ* [2] *ℓ* [2] [ .]

������ ∆ *j* = 0


**Solution to exercise 4.22 (Subset of regressors).**

1. We write ***f*** [˜] = [ . ***f*** *f* *⋆* ] *⊤* . From the definition of SoR (4.62), we gather


***u***, **0**

�


***K*** *AU* ***K*** *UU* *[−]* [1]
� ***K*** *⋆* *U* ***K*** *UU* *[−]* [1]


�


*q* SoR ( ***f*** [˜] *|* ***u*** ) = *N*


˜
***f*** ;
�


. (A.4)


We know that ***f*** [˜] and ***u*** are jointly Gaussian, and hence, the marginal
distribution of ***f*** [˜] is also Gaussian. We have for the mean and vari
ance that

˜ ˜
**E** � ***f*** � = **E** � **E** � ***f*** �� ***u*** �� using the tower rule (1.33)


***u***

�


= **E** ***u***


***K*** *AU* ***K*** *UU* *[−]* [1]
�� ***K*** *⋆* *U* ***K*** *UU* *[−]* [1] �


using eq. (A.4)


�


=


***K*** *AU* ***K*** *UU* *[−]* [1]
� ***K*** *⋆* *U* ***K*** *UU* *[−]* [1]


**E** [ ***u*** ] = **0** using linearity of expectation (1.24)


Var� ***f*** ˜� = **E** Var � ***f*** ˜ � � ***u*** �

� �� �
**0**


˜
+ Var **E** � ***f*** �� ***u*** � using the law of total variance (1.50)


�


***u***

�


= Var ***u***


***K*** *AU* ***K*** *UU* *[−]* [1]
�� ***K*** *⋆* *U* ***K*** *UU* *[−]* [1]


using eq. (A.4)

using eq. (1.47)


�


Var [ ***u*** ]
����
***K*** *UU*


� *⊤*


***K*** *AU* ***K*** *UU* *[−]* [1]
� ***K*** *⋆* *U* ***K*** *UU* *[−]* [1]


=

=


***K*** *AU* ***K*** *UU* *[−]* [1]
� ***K*** *⋆* *U* ***K*** *UU* *[−]* [1]


***Q*** *AA* ***Q*** *A* *⋆*
� ***Q*** *⋆* *A* ***Q*** *⋆⋆* �


.


Having determined *q* SoR ( ***f***, *f* *[⋆]* ), *q* SoR ( *f* *[⋆]* *|* ***y*** ) follows directly using
the formulas for finding the Gaussian process predictive posterior
(4.6).
2. The given covariance function follows directly from inspecting the
derived covariance matrix, Var [ ***f*** ] = ***Q*** *AA* .

solutions 285

*Variational Inference*

**Solution to exercise 5.4 (Logistic loss).**

1. We have

***∇*** ***w*** *ℓ* log ( ***w*** *[⊤]* ***x*** ; *y* ) = ***∇*** ***w*** log ( 1 + exp ( *−* *y* ***w*** *[⊤]* ***x*** )) using the definition of the logistic loss
(5.14)
= ***∇*** ***w*** log [1] [ +] [ ex] [p] [(] *[y]* ***[w]*** *[⊤]* ***[x]*** [)]

exp ( *y* ***w*** *[⊤]* ***x*** )

= ***∇*** ***w*** log ( 1 + exp ( *y* ***w*** *[⊤]* ***x*** )) *−* *y* ***w*** *[⊤]* ***x***

1
= 1 + exp ( *−* *y* ***w*** *[⊤]* ***x*** ) *[·]* [ exp] [(] *[−]* *[y]* ***[w]*** *[⊤]* ***[x]*** [)] *[ ·]* *[ y]* ***[x]*** *[ −]* *[y]* ***[x]*** using the chain rule


�


= *−* *y* ***x*** *·*


�


ex p ( *y* ***w*** *[⊤]* ***x*** )
1 *−*
1 + exp ( *y* ***w*** *[⊤]* ***x*** )


= *−* *y* ***x*** *·* *σ* ( *−* *y* ***w*** *[⊤]* ***x*** ) . using the definition of the logistic
function (5.10)
2. As suggested in the hint, we compute the first derivative of *σ*,


*σ* *[′]* ( *z* ) = . *d*


*d* *[d]*

*dz* *[σ]* [(] *[z]* [) =]


*dz* *[d]* 1 + exp1 ( *−* *z* ) using the definition of the logisticfunction (5.10)


= ( 1 + ex exp p ( *−* ( *−* *z* ) *z* )) [2] using the quotient rule

= *σ* ( *z* ) *·* ( 1 *−* *σ* ( *z* )) . using the definition of the logistic
function (5.10)
We get for the Hessian of *ℓ* log,

***H*** ***w*** *ℓ* log ( ***w*** *[⊤]* ***x*** ; *y* ) = ***D*** ***w*** ***∇*** ***w*** *ℓ* log ( ***w*** *[⊤]* ***x*** ; *y* ) using the definition of a Hessian (1.105)
and symmetry
= *−* *y* ***x D*** ***w*** *σ* ( *−* *y* ***w*** *[⊤]* ***x*** ) . using the gradient of the logistic loss
from (1)
We have ***D*** ***w*** *−* *y* ***w*** *[⊤]* ***x*** = *−* *y* ***x*** *[⊤]* and ***D*** *z* *σ* ( *z* ) = *σ* *[′]* ( *z* ), and therefore,
using the chain rule of multivariate calculus (5.20),

= *−* *y* ***x*** *·* *σ* *[′]* ( *−* *y* ***w*** *[⊤]* ***x*** ) *·* ( *−* *y* ***x*** *[⊤]* )

= ***xx*** *[⊤]* *·* *σ* *[′]* ( *−* *y* ***w*** *[⊤]* ***x*** ) . using *y* [2] = 1

Finally, recall that

*σ* ( ***w*** *[⊤]* ***x*** ) = **P** ( *y* = + 1 *|* ***x***, ***w*** ) and

*σ* ( *−* ***w*** *[⊤]* ***x*** ) = **P** ( *y* = *−* 1 *|* ***x***, ***w*** ) .

Thus,

*σ* *[′]* ( *−* *y* ***w*** *[⊤]* ***x*** ) = **P** ( *Y* *̸* = *y* *|* ***x***, ***w*** ) *·* ( 1 *−* **P** ( *Y* *̸* = *y* *|* ***x***, ***w*** ))

= **P** ( *Y* *̸* = *y* *|* ***x***, ***w*** ) *·* **P** ( *Y* = *y* *|* ***x***, ***w*** )

= ( 1 *−* **P** ( *Y* = *y* *|* ***x***, ***w*** )) *·* **P** ( *Y* = *y* *|* ***x***, ***w*** )

= *σ* *[′]* ( ***w*** *[⊤]* ***x*** ) .

286 probabilistic artificial intelligence

3. By the second-order characterization of convexity (cf. remark 1.58),
a twice differentiable function *f* is convex if and only if its Hessian
is positive semi-definite. Writing *c* = . *σ* *′* ( ***w*** *⊤* ***x*** ), we have for any

***δ*** *∈* **R** *[n]* that

***δ*** *[⊤]* ***H*** ***w*** *ℓ* log ( ***w*** *[⊤]* ***x*** ; *y* ) ***δ*** = ***δ*** *[⊤]* [�] *c* *·* ***xx*** *[⊤]* [�] ***δ*** = *c* ( ***δ*** *[⊤]* ***x*** ) [2] *≥* 0, using *c* *≥* 0 and ( *·* ) [2] *≥* 0

and hence, the logistic loss is convex in ***w*** .

**Solution to exercise 5.6 (Gaussian process classification).**

1. Using the law of total probability (1.15),

*p* ( *y* *[⋆]* = + 1 *|* ***x*** 1: *n*, *y* 1: *n*, ***x*** *[⋆]* )

= *p* ( *y* *[⋆]* = + 1 *|* *f* *[⋆]* ) *p* ( *f* *[⋆]* *|* ***x*** 1: *n*, *y* 1: *n*, ***x*** *[⋆]* ) *d f* *[⋆]*
�

= *σ* ( *f* *[⋆]* ) *p* ( *f* *[⋆]* *|* ***x*** 1: *n*, *y* 1: *n*, ***x*** *[⋆]* ) *d f* *[⋆]* . using *y* *[⋆]* *∼* Bern ( *σ* ( *f* *[⋆]* ))
�

Due to the non-Gaussian likelihood, the integral is analytically intractable. However, as the integral is one-dimensional, numerical approximations such as the Gauss-Legendre quadrature can be

used.

2. (a) According to Bayes’ rule (1.58), we know that

*ψ* ( ***f*** ) = log *p* ( ***f*** *|* ***x*** 1: *n*, *y* 1: *n* ) using eq. (5.2)

= log *p* ( *y* 1: *n* *|* ***f*** ) + log *p* ( ***f*** *|* ***x*** 1: *n* ) *−* log *p* ( *y* 1: *n* *|* ***x*** 1: *n* ) using Bayes’ rule (1.58)

= log *p* ( *y* 1: *n* *|* ***f*** ) + log *p* ( ***f*** *|* ***x*** 1: *n* ) + const.

Plugging in the closed-form Gaussian distribution of the GP
prior gives

= log *p* ( *y* 1: *n* *|* ***f*** )



[1]

*AA* ***[f]*** *[ −]* [1]
2 ***[f]*** *[ ⊤]* ***[K]*** *[−]* [1]


2 [log 2] *[π]* [ +] [ const]


*−* [1]


2 [log det] [(] ***[K]*** *[AA]* [)] *[ −]* *[n]*


Differentiating with respect to ***f*** yields

***∇*** *ψ* ( ***f*** ) = ***∇*** log *p* ( *y* 1: *n* *|* ***f*** ) *−* ***K*** *[−]* *AA* [1] ***[f]***

***H*** *ψ* ( ***f*** ) = ***H*** ***f*** log *p* ( *y* 1: *n* *|* ***f*** ) *−* ***K*** *[−]* *AA* [1] [=] *[ −]* ***[W]*** *[ −]* ***[K]*** *[−]* *AA* [1]

where ***W*** = . *−* ***H*** ***f*** log *p* ( *y* 1: *n* *|* ***f*** ) . Hence, ***Λ*** = ***K*** *−* *AA* 1 [+] ***[ W]*** [.]
It remains to derive ***W*** . Using independence of the training
examples, log *p* ( *y* 1: *n* *|* ***f*** ) = ∑ *i* *[n]* = 1 [log] *[ p]* [(] *[y]* *[i]* *[ |]* *[ f]* *[i]* [)] [, and hence, the]
Hessian of this expression is diagonal. Using the symmetry of
Φ ( *z* ; 0, *σ* *n* [2] [)] [ around zero, we can write]

log *p* ( *y* *i* *|* *f* *i* ) = log Φ ( *y* *i* *f* *i* ; 0, *σ* *n* [2] [)] [.]

solutions 287


In the following, we write *N* ( *z* ) = . *N* ( *z* ; 0, *σ* *n* 2 [)] [ and][ Φ] [(] *[z]* [)] = .
Φ ( *z* ; 0, *σ* *n* [2] [)] [ to simplify the notation. Differentiating with respect]
to *f* *i*, we obtain

*∂* log ( *y* *i* *f* *i* ) = *[y]* *[i]* *[N]* [(] *[f]* *[i]* [)]
*∂* *f* *i* Φ ( *y* *i* *f* *i* )


*∂* [2]



*[N]* [(] *[f]* *[i]* [)] [2] [(] *[f]* *[i]* [)]

Φ ( *y* *i* *f* *i* ) [2] *[ −]* *σ* *[y]* *n* *[i]* [2] *[ f]* Φ *[i]* *[N]* ( *y* *i* *f* *i* )


*∂* [2] [(] *[f]* *[i]* [)] [2]

log Φ ( *y* *i* *f* *i* ) = *−* *[N]*
*∂* *f* *i* [2] Φ ( *y* *i* *f* *i* )


*σ* *n* [2] Φ ( *y* *i* *f* *i* ) [,]


and ***W*** = *−* diag *{* *∂* *[∂]* *f* [2] *i* [2] [log][ Φ] [(] *[y]* *[i]* *[ f]* *[i]* [)] *[}]* *i* *[n]* = 1 [.]


(b) Similarly to eq. (4.6), evaluating a GP posterior at a test point
***x*** *[⋆]* yields,

*f* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, ***f*** *∼N* ( *µ* *[⋆]*, *k* *[⋆]* ), where (A.5a)

*µ* *[⋆]* = . ***k*** *⊤* ***x*** *[⋆]*, *A* ***[K]*** *[−]* *AA* [1] ***[f]*** [,] (A.5b)

*k* *[⋆]* = . *k* ( ***x*** *⋆*, ***x*** *⋆* ) *−* ***k*** *⊤* ***x*** *[⋆]*, *A* ***[K]*** *[−]* *AA* [1] ***[k]*** ***[x]*** *[⋆]* [,] *[A]* [.] (A.5c)

Then, using the tower rule (1.33),

**E** *q* [ *f* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ] = **E** *q* [ **E** [ *f* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, ***f*** ] *|* ***x*** 1: *n*, *y* 1: *n* ]

= ***k*** *[⊤]* ***x*** *[⋆]*, *A* ***[K]*** *[−]* *AA* [1] **[E]** *[q]* [ [] ***[ f]*** *[ |]* ***[ x]*** [1:] *[n]* [,] *[ y]* [1:] *[n]* []] using eq. (A.5b) and linearity of
expectation (1.24)
= ***k*** *[⊤]* ***x*** *[⋆]*, *A* ***[K]*** *[−]* *AA* [1] ***[f]*** [ˆ][.]

Observe that any maximum ***f*** [ˆ] of *ψ* ( ***f*** ) needs to satisfy ***∇*** *ψ* ( ***f*** ) =
**0** . Hence, ***f*** [ˆ] = ***K*** *AA* ( ***∇*** ***f*** log *p* ( *y* 1: *n* *|* ***f*** )), and the expectation
simplifies to

= ***k*** *[⊤]* ***x*** *[⋆]*, *A* [(] ***[∇]*** ***[f]*** [log] *[ p]* [(] *[y]* [1:] *[n]* *[|]* ***[ f]*** [))] [.]

Using the law of total variance (1.50),

Var *q* [ *f* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, *y* 1: *n* ]

= **E** ***f*** *∼* *q* �Var *f* *⋆* *∼* *p* [ *f* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, ***f*** ] ��� ***x*** 1: *n*, *y* 1: *n* �

+ Var ***f*** *∼* *q* � **E** *f* *⋆* *∼* *p* [ *f* *[⋆]* *|* ***x*** *[⋆]*, ***x*** 1: *n*, ***f*** ] ��� ***x*** 1: *n*, *y* 1: *n* �

= **E** *q* [ *k* *[⋆]* *|* ***x*** 1: *n*, *y* 1: *n* ] + Var *q* � ***k*** *[⊤]* ***x*** *[⋆]*, *A* ***[K]*** *[−]* *AA* [1] ***[f]*** ��� ***x*** 1: *n*, *y* 1: *n* � using eqs. (A.5b) and (A.5c)

= *k* *[⋆]* + ***k*** *[⊤]* ***x*** *[⋆]*, *A* ***[K]*** *[−]* *AA* [1] [Var] *[q]* [ [] ***[ f]*** *[ |]* ***[ x]*** [1:] *[n]* [,] *[ y]* [1:] *[n]* []] ***[ K]*** *[−]* *AA* [1] ***[k]*** ***[x]*** *[⋆]* [,] *[A]* [.] using that *k* *[⋆]* is independent of ***f***,

eq. (1.47), and symmetry of ***K*** *[−]* *AA* [1]
Recall from (a) that Var *q* [ ***f*** *|* ***x*** 1: *n*, *y* 1: *n* ] = ( ***K*** *AA* + ***W*** *[−]* [1] ) *[−]* [1], so


= *k* ( ***x*** *[⋆]*, ***x*** *[⋆]* ) *−* ***k*** *[⊤]* ***x*** *[⋆]*, *A* ***[K]*** *[−]* *AA* [1] ***[k]*** ***[x]*** *[⋆]* [,] *[A]*
+ ***k*** *[⊤]* ***x*** *[⋆]*, *A* ***[K]*** *[−]* *AA* [1] [(] ***[K]*** *[AA]* [ +] ***[ W]*** *[−]* [1] [)] *[−]* [1] ***[K]*** *[−]* *AA* [1] ***[k]*** ***[x]*** *[⋆]* [,] *[A]*


plugging in the expression for *k* *[⋆]* (A.5c)


= *k* ( ***x*** *[⋆]*, ***x*** *[⋆]* ) *−* ***k*** *[⊤]* ***x*** *[⋆]*, *A* [(] ***[K]*** *[AA]* [+] ***[ W]*** *[−]* [1] [)] *[−]* [1] ***[k]*** ***[x]*** *[⋆]* [,] *[A]* [.] using the matrix inversion lemma (5.26)


Note that the (approximate) latent predictive posterior that uses
the Laplace-approximated latent posterior is a Gaussian, [2] and

hence, is described fully by its mean and variance.


2 Observe that using the Laplaceapproximated latent posterior, [ *f* *[⋆]* ***f*** ]
are jointly Gaussian. Thus, it directly
follows from theorem 1.69 that the
marginal distribution over *f* *[⋆]* is also a
Gaussian.

288 probabilistic artificial intelligence

(c) Recall that

*p* ( *y* *[⋆]* = + 1 *|* ***x*** 1: *n*, *y* 1: *n*, ***x*** *[⋆]* ) *≈* *σ* ( *f* *[⋆]* ) *q* ( *f* *[⋆]* *|* ***x*** 1: *n*, *y* 1: *n*, ***x*** *[⋆]* ) *d f* *[⋆]* using the Laplace-approximated latent
�
predictive posterior

= **E** *q* [ *σ* ( *f* *[⋆]* )] . using LOTUS (1.30)

This quantity can be interpreted as the *averaged prediction* over
all latent predictions *f* *[⋆]* . In contrast, *σ* ( **E** *q* [ *f* *[⋆]* ]) can be understood as the *MAP prediction*, which is obtained using the MAP
estimate of *f* *[⋆]* . [3] As *σ* is non-linear, the two quantities are not 3 As *q* is a Gaussian, its mode (i.e., the
identical, and generally the averaged prediction is preferred. MAP estimate) and its mean coincide.
3. Given the latent variable *f* *i*, we write *f* [˜] *i* = . *f* *i* + *ϵ* *i* with noise *ϵ* *i* *∼*
*N* ( 0, *σ* *n* [2] [)] [. We want to show that] **[ E]** *[ϵ]* *i* � *h* ( *f* [˜] *i* ) � = Φ ( *f* *i* ; 0, *σ* *n* 2 [)] [.]
It follows from the use of the step-function likelihood (5.27) that
*h* ( *f* [˜] *i* ) *∼* Bern ( **P** � *f* ˜ *i* *≥* 0� ), and therefore,

˜
**E** � *h* ( *f* [˜] *i* ) � = **P** � *f* *i* *≥* 0� using **E** [ *X* ] = *p* if *X* *∼* Bern ( *p* )

= **P** ( *−* *ϵ* *i* *≤* *f* *i* )

= **P** ( *ϵ* *i* *≤* *f* *i* ) using that the distribution of *ϵ* *i* is
symmetric around 0
= Φ ( *f* *i* ; 0, *σ* *n* [2] [)] [.]

**Solution to exercise 5.16 (Binary cross-entropy loss).** If *y* = 1 then

*ℓ* bce ( ˆ *y* ; *y* ) = *−* log ˆ *y* = log ( 1 + *e* *[−]* *[f]* [ˆ] ) = *ℓ* log ( *F* [ˆ] ; *y* ) .

If *y* = *−* 1 then

ˆ
*ℓ* bce ( ˆ *y* ; *y* ) = *−* log ( 1 *−* *y* ˆ ) = log ( 1 + *e* *f* ) = *ℓ* log ( ˆ *f* ; *y* ) .

Here the second equality follows from the simple algebraic fact


1 *e* *[z]*
1 *−*
1 + *e* *[−]* *[z]* [ =] [ 1] *[ −]* *e* *[z]*


*e* *[z]* 1

*e* *[z]* + 1 [=] 1 + *e* *[z]* [ .] multiplying by *[e]* *e* *[z][z]*


*e* *[z]*


**Solution to exercise 5.11 (Jensen’s inequality).**

1. Recall that as *f* is convex,

*∀* ***x*** 1, ***x*** 2, *∀* *λ* *∈* [ 0, 1 ] : *f* ( *λ* ***x*** 1 + ( 1 *−* *λ* ) ***x*** 2 ) *≤* *λ f* ( ***x*** 1 ) + ( 1 *−* *λ* ) *f* ( ***x*** 2 ) .

We prove the statement by induction on *k* . The base case, *k* = 2,
follows trivially from the convexity of *f* . For the induction step,
suppose that the statement holds for some fixed *k* *≥* 2 and assume
w.l.o.g. that *θ* *k* + 1 *∈* ( 0, 1 ) . We then have,


*k* + 1
###### ∑ θ i f ( x i ) = ( 1 − θ k + 1 )

*i* = 1


*k*
###### ∑
� *i* = 1


*θ* *i*
*f* ( ***x*** *i* )
1 *−* *θ* *k* + 1


�

�


+ *θ* *k* + 1 *f* ( ***x*** *k* + 1 )

+ *θ* *k* + 1 *f* ( ***x*** *k* + 1 ) using the induction hypothesis


*≥* ( 1 *−* *θ* *k* + 1 ) *·* *f*


*k*
###### ∑
� *i* = 1


*θ* *i*
***x*** *i*
1 *−* *θ* *k* + 1

solutions 289

. using the convexity of *f*

�


*≥* *f*


*k* + 1
###### ∑ θ i x i
� *i* = 1


2. Noting that log 2 is concave, we have by Jensen’s inequality,


1
� *p* ( *x* )


��


H [ *p* ] = **E** *x* *∼* *p*


log 2
�


�


*≤* log 2 **E** *x* *∼* *p*

= log 2 *n* .


1
� *p* ( *x* )


**Solution to exercise 5.20 (Gibbs’ inequality).**

1. Let *p* and *q* be two (continuous) distributions. The KL-divergence
between *p* and *q* is


KL ( *p* *∥* *q* ) = **E** ***x*** *∼* *p*

= **E** ***x*** *∼* *p*


log *[p]* [(] ***[x]*** [)]
� *q* ( ***x*** )


*q* ( ***x*** )
S
� � *p* ( ***x*** )


.
��


�


Note that the surprise S [ *u* ] = *−* log *u* is a convex function in *u*, and
hence,


by definition of entropy (5.30a)

by Jensen’s inequality (5.34)

using the definition of KL-divergence
(5.43)

using Jensen’s inequality (5.33)


*q* ( ***x*** )

*≥* S **E** ***x*** *∼* *p*
� � *p* ( ***x*** )


��


= S *q* ( ***x*** ) *d* ***x***
� � �

= S [ 1 ] = 0. a probability density integrates to 1

2. We observe from the derivation of (1) that KL ( *p* *∥* *q* ) = 0 iff equality
holds for Jensen’s inequality. Now, if *p* and *q* are discrete with final
and identical support, we can follow from the hint that Jensen’s
inequality degenerates to an equality iff *p* and *q* are point wise

identical.

**Solution to exercise 5.21 (Maximum entropy principle).**

1. We have


H [ *f* *∥* *g* ] = *−*
�

= *−*
�


1
**R** *[f]* [ (] *[x]* [)] *[ ·]* � log� *√* 2 *πσ* [2]


**R** *[f]* [ (] *[x]* [)] [ log] *[ g]* [(] *[x]* [)] *[ dx]* using the definition of cross-entropy(5.37)


*dx* using that *g* ( *x* ) = *N* ( *x* ; *µ*, *σ* [2] )
�


*−* [(] *[x]* *[ −]* *[µ]* [)] [2]
� 2 *σ* [2]


2 *σ* [2]


1
= *−* log
� *√* 2 *πσ* [2]


� � � **R** *[f]* [ (] �� *[x]* [)] *[ dx]* �

1


+ [1]

2 *σ* [2]


� **R** *[f]* [ (] *[x]* [)(] *[x]* *[ −]* *[µ]* [)] [2] *[ dx]*


= log ( *σ* *√*


1
2 *π* ) + ( *x* *−* *µ* ) [2] [�]
2 *σ* [2] **[ E]** *[x]* *[∼]* *[f]* �

� �� �
*σ* [2]

290 probabilistic artificial intelligence


= log ( *σ* *√*


2 *π* ) + [1]


2

= H [ *g* ] . using the entropy of Gaussians (5.31)


2. We have shown that

H [ *g* ] *−* H [ *f* ] = KL ( *f* *∥* *g* ) *≥* 0,

and hence, H [ *g* ] *≥* H [ *f* ] . That is, for a fixed mean *µ* and variance
*σ* [2], the distribution that has maximum entropy among all distributions that are supported on **R** is the normal distribution.

**Solution to exercise 5.24 (KL-divergence of Gaussians).** We can rewrite
the KL-divergence as


KL ( *p* *∥* *q* ) = **E** ***x*** *∼* *p* [ log *p* ( ***x*** ) *−* log *q* ( ***x*** )] using the definition of KL-divergence
(5.43)


= **E** ***x*** *∼* *p*


1 � ***Σ*** *q* �

[det]
� 2 [log] det� ***Σ*** *p* �



[det] � ***Σ*** *q* � *−* [1]

det� ***Σ*** *p* � 2


2 [(] ***[x]*** *[ −]* ***[µ]*** *[p]* [)] *[⊤]* ***[Σ]*** *[−]* *p* [1] [(] ***[x]*** *[ −]* ***[µ]*** *[p]* [)]


+ [1] 2 [(] ***[x]*** *[ −]* ***[µ]*** *[q]* [)] *[⊤]* ***[Σ]*** *[−]* *q* [1] [(] ***[x]*** *[ −]* ***[µ]*** *[q]* [)] �


using the Gaussian PDF (1.114)

using linearity of expectation (1.24)


� ***Σ*** *q* �

[1] [det]

2 [log] det ***Σ***


2 **[E]** ***[x]*** *[∼]* *[p]* � ( ***x*** *−* ***µ*** *p* ) *[⊤]* ***Σ*** *[−]* *p* [1] [(] ***[x]*** *[ −]* ***[µ]*** *[p]* [)] �


= [1]



[det] � ***Σ*** *q* � *−* [1]

det� ***Σ*** *p* � 2


+ [1] 2 **[E]** ***[x]*** *[∼]* *[p]* � ( ***x*** *−* ***µ*** *q* ) *[⊤]* ***Σ*** *[−]* *q* [1] [(] ***[x]*** *[ −]* ***[µ]*** *[q]* [)] �


As ( ***x*** *−* ***µ*** *p* ) *[⊤]* ***Σ*** *[−]* *p* [1] [(] ***[x]*** *[ −]* ***[µ]*** *[p]* [)] *[ ∈]* **[R]** [, we can rewrite the second term as]

1
2 **[E]** ***[x]*** *[∼]* *[p]* �tr� ( ***x*** *−* ***µ*** *p* ) *[⊤]* ***Σ*** *[−]* *p* [1] [(] ***[x]*** *[ −]* ***[µ]*** *[p]* [)] ��

= 2 [1] **[E]** ***[x]*** *[∼]* *[p]* �tr� ( ***x*** *−* ***µ*** *p* )( ***x*** *−* ***µ*** *p* ) *[⊤]* ***Σ*** *[−]* *p* [1] �� using the cyclic property of the trace


= [1]


= 2 [tr] � **E** ***x*** *∼* *p* � ( ***x*** *−* ***µ*** *p* )( ***x*** *−* ***µ*** *p* ) *[⊤]* [�] ***Σ*** *[−]* *p* [1] � using linearity of the trace and linearity

of expectation (1.24)
= [1] ***Σ*** ***Σ*** *[−]* [1]

[tr]


= 2 [tr] � ***Σ*** *p* ***Σ*** *[−]* *p* [1] � using the definition of the covariance

matrix (1.44)
= [1] [tr] ***[I]*** *[d]* [.]



[1] *[d]*

2 [tr] [(] ***[I]*** [) =] 2


2 [.]


For the third term, we use the hint (5.51) to obtain

1
2 **[E]** ***[x]*** *[∼]* *[p]* � ( ***x*** *−* ***µ*** *q* ) *[⊤]* ***Σ*** *[−]* *q* [1] [(] ***[x]*** *[ −]* ***[µ]*** *[q]* [)] �


= [1]

2


� ( ***µ*** *p* *−* ***µ*** *q* ) *[⊤]* ***Σ*** *[−]* *q* [1] [(] ***[µ]*** *[p]* *[−]* ***[µ]*** *[q]* [) +] [ tr] � ***Σ*** *[−]* *q* [1] ***[Σ]*** *[p]* ��.


Putting all terms together we get


log det [det] � � ***ΣΣ*** *q* *p* � � *−* *d* + ( ***µ*** *p* *−* ***µ*** *q* ) *[⊤]* ***Σ*** *[−]* *q* [1] [(] ***[µ]*** *[p]* *[−]* ***[µ]*** *[q]* [)]

+ tr� ***Σ*** *[−]* *q* [1] ***[Σ]*** *[p]* �� .


KL ( *p* *∥* *q* ) = [1]

2


�

solutions 291

**Solution to exercise 5.27 (Forward vs reverse KL).**

1. Let *p* and *q* be discrete distributions. The derivation is analogous
if *p* and *q* are taken to be continuous. First, we write the KLdivergence between *p* and *q* as

KL ( *p* *∥* *q* )
###### = ∑ p ( x, y ) log 2 p ( x, y ) using the definition of KL-divergence x [∑] y q ( x ) q ( y ) (5.43) = ∑ x [∑] y p ( x, y ) log 2 p ( x, y ) − ∑ x [∑] y p ( x, y ) log 2 q ( x ) − ∑ x [∑] y p ( x, y ) log 2 q ( y ) = ∑ x [∑] y p ( x, y ) log 2 p ( x, y ) − ∑ x p ( x ) log 2 q ( x ) − ∑ y p ( y ) log 2 q ( y ) using the sum rule (1.10)

= *−* H [ *p* ( *x*, *y* )] + H [ *p* ( *x* ) *∥* *q* ( *x* )] + H [ *p* ( *y* ) *∥* *q* ( *y* )] using the definitions of entropy (5.29)
and cross-entropy (5.37)
= *−* H [ *p* ( *x*, *y* )] + H [ *p* ( *x* )] + H [ *p* ( *y* )] + KL ( *p* ( *x* ) *∥* *q* ( *x* )) using eq. (5.38)

+ KL ( *p* ( *y* ) *∥* *q* ( *y* ))

= KL ( *p* ( *x* ) *∥* *q* ( *x* )) + KL ( *p* ( *y* ) *∥* *q* ( *y* )) + const.

Hence, to minimize KL ( *p* *∥* *q* ) with respect to the variational distributions *q* ( *x* ) and *q* ( *y* ) we should set KL ( *p* ( *x* ) *∥* *q* ( *x* )) = 0 and
KL ( *p* ( *y* ) *∥* *q* ( *y* )) = 0, respectively. This is obtained when

*q* ( *x* ) = *p* ( *x* ) and *q* ( *y* ) = *p* ( *y* ) .

2. The reverse KL-divergence KL ( *q* *∥* *p* ) on the finite domain *x*, *y* *∈*
*{* 1, 2, 3, 4 *}* is defined as

*q* ( *x* ) *q* ( *y* )
###### KL ( q ∥ p ) = ∑ q ( x ) q ( y ) log 2 x [∑] y p ( x, y ) [.]


We can easily observe from the above formula that the support
of *q* must be a subset of the support of *p* . In other words, if
*q* ( *x*, *y* ) is positive outside the support of *p* (i.e., when *p* ( *x*, *y* ) = 0)
then KL ( *q* *∥* *p* ) = ∞. Hence, the reverse KL-divergence has an infinite value except when the support of *q* is either *{* 1, 2 *} × {* 1, 2 *}* or
*{* ( 3, 3 ) *}* or *{* ( 4, 4 ) *}* . Thus, it has three local minima.
For the first case, the minimum is achieved when *q* ( *x* ) = *q* ( *y* ) =
( [1] 2 [,] 2 [1] [, 0, 0] [)] [. The corresponding KL-divergence is KL] [(] *[q]* *[∥]* *[p]* [) =] [ log] 2 [2]



[1] [1]

2 [,] 2


( 2 [,] 2 [, 0, 0] [)] [. The corresponding KL-divergence is KL] [(] *[q]* *[∥]* *[p]* [) =] [ log] 2 [2] [ =]

1. For the second case and the third case, *q* ( *x* ) = *q* ( *y* ) = ( 0, 0, 1, 0 )
and *q* ( *x* ) = *q* ( *y* ) = ( 0, 0, 0, 1 ), respectively. The KL-divergence in
both cases is KL ( *q* *∥* *p* ) = log 2 4 = 2.
3. Let us compute *p* ( *x* = 4 ) and *p* ( *y* = 1 ) :

###### p ( x = 4 ) = ∑ p ( x = 4, y ) = [1]

4 [,]

*y*

292 probabilistic artificial intelligence
###### p ( y = 1 ) = ∑ p ( x, y = 1 ) = [1]

*x* 4 [.]

1
Hence, *q* ( *x* = 4, *y* = 1 ) = *p* ( *x* = 4 ) *p* ( *y* = 1 ) = 16 [, however,] *[ p]* [(] *[x]* [ =]
4, *y* = 1 ) = 0. We therefore have for the reverse KL-divergence that
KL ( *q* *∥* *p* ) = ∞.

**Solution to exercise 5.34 (Gradient of reverse-KL).** To simplify the
notation, we write ***Σ*** = . diag *{* *σ* 12 [, . . .,] *[ σ]* *d* [2] *[}]* [. The reverse KL-divergence]
can be expressed as


�


using the expression for the
KL-divergence of Gaussians (5.50)


( *σ* *p* [2] [)] *[d]*
tr� *σ* *p* *[−]* [2] ***[Σ]*** � + *σ* *p* *[−]* [2] ***[µ]*** *[⊤]* ***[µ]*** *[ −]* *[d]* [ +] [ log] det ( ***Σ*** )


KL ( *q* ***λ*** *∥* *p* ( *·* )) = [1]

2


.

�


= [1]

2


�

�


*σ* *[−]* [2]
*p*


*d* *d*
###### ∑ σ i [2] [+] [ σ] p [−] [2] [µ] [⊤] [µ] [ −] [d] [ +] [ d] [ log] [ σ] p [2] [−] ∑ log σ i [2]

*i* = 1 *i* = 1


It follows immediately that ***∇*** ***µ*** KL ( *q* ***λ*** *∥* *p* ( *·* )) = *σ* *p* *[−]* [2] ***[µ]*** [. Moreover,]


*−* *∂σ* *[∂]* *i* log *σ* *i* [2]

� �� �
2 / *σi*


*∂*
KL ( *q* ***λ*** *∥* *p* ( *·* )) = [1]
*∂σ* *i* 2


*∂*
� *σ* *p* *[−]* [2] *∂σ* *i* *σ* *i* [2]
����
2 *σ* *i*



*[σ]* *[i]* *−* [1]

*σ* [2] *σ*
*p*


.
*σ* *i*


= *[σ]* *[i]*
� *σ*


**Solution to exercise 5.37 (Reparameterizable distributions).**

1. Let *Y* *∼* Unif ([ 0, 1 ]) . Then, using the two hints,

( *b* *−* *a* ) *Y* + *a* *∼* Unif ([ *a*, *b* ]) .

2. Let *Z* 1 *∼N* ( *µ*, *σ* [2] ) and *Z* 2 *∼N* ( 0, 1 ) . We have, *X* = *e* *[Z]* [1] . Recall
from eq. (1.122) that *Z* 1 can equivalently be expressed in terms of
*Z* 2 as *Z* 1 = *σZ* 2 + *µ* . This yields,

*X* = *e* *[Z]* [1] = *e* *[σ][Z]* [2] [+] *[µ]* .

*Markov Chain Monte Carlo Methods*

**Solution to exercise 6.3 (Markov chain update).** We have

###### q t + 1 ( x [′] ) = P � X t + 1 = x [′] [�] = ∑ P ( X t = x )

*x* � �� �
*q* *t* ( *x* )


**P** � *X* *t* + 1 = *x* *[′]* *|* *X* *t* = *x* �

� �� �
*p* ( *x* *[′]* *|* *x* )


. using the sum rule (1.10)


Noting that *p* ( *x* *[′]* *|* *x* ) = ***P*** ( *x*, *x* *[′]* ), we conclude ***q*** *t* + 1 = ***q*** *t* ***P*** .

**Solution to exercise 6.4 (** *k* **-step transitions).** It follows directly from
the definition of matrix multiplication that
###### P [k] ( x, x [′] ) = ∑ P ( x, x 1 ) · P ( x 1, x 2 ) · · · P ( x k − 1, x [′] )

*x* 1,..., *x* *k* *−* 1

solutions 293
###### = ∑ P ( X 1 = x 1 | X 0 = x ) · · · P � X k = x [′] | X k − 1 = x k − 1 � using the definition of the transition

*x* 1,..., *x* *k* *−* 1 matrix (6.8)
###### = ∑ P � X 1 = x 1, · · ·, X k − 1 = x k − 1, X k = x [′] | X 0 = x � using the product rule (1.14)

*x* 1,..., *x* *k* *−* 1

= **P** � *X* *k* = *x* *[′]* *|* *X* 0 = *x* �. using the sum rule (1.10)

**Solution to exercise 6.14 (Finding stationary distributions).** We con
sider the transition matrix




.



***P*** =


0.60 0.30 0.10

0.50 0.25 0.25

0.20 0.40 0.40



We note that the entries of ***P*** are all different from 0, thus the Markov
chain corresponding to this transition matrix is ergodic. [4] Thus, there 4 All elements of the transition matrix be
exists a unique stationary distribution *π* to which the Markov chain ing strictly greater than 0 is a sufficient,

but not necessary, condition for ergodic
converges irrespectively of the distribution over initial states *q* 0 . ity.

We know that ***P*** *[⊤]* ***π*** = ***π*** (where we write ***π*** as a column vector),
therefore, to find the stationary distribution *π*, we need to find the
normalized eigenvector associated with eigenvalue 1 of the matrix ***P*** *[⊤]* .
That is, we want to solve ( ***P*** *[⊤]* *−* ***I*** ) ***π*** = **0** for ***π*** . We obtain the linear
system of equations,

*−* 0.40 *π* 1 + 0.50 *π* 2 + 0.20 *π* 3 = 0

0.30 *π* 1 *−* 0.75 *π* 2 + 0.40 *π* 3 = 0

0.10 *π* 1 + 0.25 *π* 2 *−* 0.60 *π* 3 = 0.

Note that the left hand side of equation *i* corresponds to the probability of entering state *i* at stationarity minus *π* *i* . Quite intuitively, this
difference should be 0, that is, after one iteration the random walk is

at state *i* with the same probability as before the iteration.

Solving this system of equations, for example, using the Gaussian
elimination algorithm, we obtain the normalized eigenvector




.



***π*** = [1]

72


35

22

15



Thus, we conclude that in the long run, the percentage of news days
that will be classified as “good” is [35] / 72 .

**Solution to exercise 6.23 (Gibbs sampling).** We have to compute the
conditional distributions. Notice that for *x* *∈{* 0, . . ., *n* *}* and *y* *∈* [ 0, 1 ],


*n*
*p* ( *x*, *y* ) =
� *x*


*y* *[x]* ( 1 *−* *y* ) *[n]* *[−]* *[x]* *·* *y* *[α]* *[−]* [1] ( 1 *−* *y* ) *[β]* *[−]* [1] = Bin ( *x* ; *n*, *y* ) *·* *C* *y*,
�

294 probabilistic artificial intelligence

where Bin ( *n*, *y* ) is the PMF of the binomial distribution (1.65) with *n*
trials and success probability *y*, and *C* *y* is a constant depending on *y* .
It is clear that

*p* ( *x* *|* *y* ) = *[p]* [(] *[x]* [,] *[y]* [)] using the definition of conditional

*p* ( *y* ) probability (1.11)

*[y]* [)] *[ ·]* *[ C]* *[y]*
= [Bin] [(] *[x]* [;] *[ n]* [,]

*p* ( *y* )

= Bin ( *x* ; *n*, *y* ) . using that *p* ( *x* *|* *y* ) is a probability
distribution over *x* and Bin ( *x* ; *n*, *y* )
So in short, sampling from *p* ( *x* *|* *y* ) is equivalent to sampling from a already sums to 1, so *C* *y* = *p* ( *y* )
binomial distribution, which amounts to *n* times throwing a coin with
bias *y*, and outputting the number of heads.

For the other conditional distribution, recall the PMF of the beta dis
tribution with parameters *α*, *β*,

Beta ( *y* ; *α*, *β* ) = *C* *·* *y* *[α]* *[−]* [1] ( 1 *−* *y* ) *[β]* *[−]* [1]

where *C* is some constant depending on *α* and *β* only. We then have

*p* ( *x*, *y* ) = Beta ( *y* ; *x* + *α*, *n* *−* *x* + *β* ) *·* *C* *x*,

where *C* *x* is some constant depending on *x*, *α*, *β* . This shows (analogously to above), that

*p* ( *y* *|* *x* ) = Beta ( *y* ; *x* + *α*, *n* *−* *x* + *β* ) .

So for sampling *y* given *x*, one can sample from the beta distribution.
There are several methods for sampling from a beta distribution, and
[we refer the reader to the corresponding Wikipedia page.](https://en.wikipedia.org/wiki/Beta_distribution)

**Solution to exercise 6.25 (Maximum entropy property of Gibbs dis-**
**tribution).**

1. Our goal is to solve the optimization problem

max *p* ( *x* ) log 2 *p* ( *x* )
###### p ∈ ∆ [T] [ −] x [∑] ∈T

(A.6)
###### subject to ∑ p ( x ) f ( x ) = µ

*x* *∈T*

for some *µ* *∈* **R** . The Lagrangian with dual variables *λ* 0 and *λ* 1 is
given by


�

�

###### L ( p, λ 0, λ 1 ) = − ∑ p ( x ) log 2 p ( x ) + λ 0

*x* *∈T*

###### 1 − ∑ p ( x )

*x* *∈T*


�


+


�


*λ* 1

###### µ − ∑ p ( x ) f ( x )

*x* *∈T*

solutions 295
###### = − ∑ p ( x )( log 2 p ( x ) + λ 0 + λ 1 f ( x )) + const.

*x* *∈T*

Note that *L* ( *p*, *λ* 0, *λ* 1 ) = H [ *p* ] if the constraints are satisfied. Thus,
we simply need to solve the (dual) optimization problem

max min
***p*** *≥* **0** *λ* 0, *λ* 1 *∈* **R** *[L]* [(] ***[p]*** [,] *[ λ]* [0] [,] *[ λ]* [1] [)] [.]

We have

*∂*
*∂p* ( *x* ) *[L]* [(] *[p]* [,] *[ λ]* [0] [,] *[ λ]* [1] [) =] *[ −]* [log] [2] *[ p]* [(] *[x]* [)] *[ −]* *[λ]* [0] *[ −]* *[λ]* [1] *[ f]* [ (] *[x]* [)] *[ −]* [1.]

Setting the partial derivatives to zero, we obtain

*p* ( *x* ) = 2 exp ( *−* *λ* 0 *−* *λ* 1 *f* ( *x* ) *−* 1 ) using log 2 ( *·* ) = log [lo] [g] ( [(] 2 *[·]* [)] )

∝ exp ( *−* *λ* 1 *f* ( *x* )) .

Clearly, *p* is a valid probability mass function when normalized
(i.e., for an appropriate choice of *λ* 0 ). We complete the proof by
setting *T* = . *λ* 1 1 [.]
2. As *T* *→* ∞ (and *λ* 1 *→* 0), the optimization problem reduces to picking the maximum entropy distribution without the first-moment
constraint. This distribution is the uniform distribution over *T* .

Conversely, as *T* *→* 0 (and *λ* 1 *→* ∞), the Gibbs distribution reduces to a point density around its mode.

**Solution to exercise 6.26 (Energy function of Bayesian logistic regres-**
**sion).** First, note that the sum of convex functions is convex, hence,

we consider each term individually.

The Hessian of the regularization term is *λ* ***I***, and thus, by the secondorder characterization of convexity, this term is convex in ***w*** .

Finally, note that the second term is a sum of logistic losses *ℓ* log (5.14),
and we have seen in exercise 5.4 that *ℓ* log is convex in ***w*** .

**Solution to exercise 6.27 (Energy reduction of Gibbs sampling).** Recall the following two facts:

1. Gibbs sampling is an instance of Metropolis-Hastings with proposal distribution


0 otherwise


*r* ( ***x*** *[′]* *|* ***x*** ) =






0 *p* ( *x* *i* *[′]* *[|]* ***[ x]*** *[′−]* *[i]* [)] ifotherwise ***x*** *[′]* differs from ***x*** only in entry *i*




and acceptance distribution *α* ( ***x*** *[′]* *|* ***x*** ) = 1.

296 probabilistic artificial intelligence

2. The acceptance distribution of Metropolis-Hastings where the stationary distribution *p* is a Gibbs distribution with energy function
*f* is

*α* ( ***x*** *[′]* *|* ***x*** ) = min 1, *[r]* [(] ***[x]*** *[|]* ***[ x]*** *[′]* [)] .
� *r* ( ***x*** *[′]* *|* ***x*** ) [exp] [(] *[ f]* [ (] ***[x]*** [)] *[ −]* *[f]* [ (] ***[x]*** *[′]* [))] �

We therefore know that

*r* ( ***x*** *|* ***x*** *[′]* )
*r* ( ***x*** *[′]* *|* ***x*** ) [exp] [(] *[ f]* [ (] ***[x]*** [)] *[ −]* *[f]* [ (] ***[x]*** *[′]* [))] *[ ≥]* [1.]

We remark that this inequality even holds with equality using our
derivation of theorem 6.21. Taking the logarithm and reorganizing the
terms, we obtain

*f* ( ***x*** *[′]* ) *≤* *f* ( ***x*** ) + log *r* ( ***x*** *|* ***x*** *[′]* ) *−* log *r* ( ***x*** *[′]* *|* ***x*** ) . (A.7)

By the definition of the proposal distribution of Gibbs sampling,

*r* ( ***x*** *[′]* *|* ***x*** ) = *p* ( *x* *i* *[′]* *[|]* ***[ x]*** *[−]* *[i]* [)] and *r* ( ***x*** *|* ***x*** *[′]* ) = *p* ( *x* *i* *|* ***x*** *−* *i* ) . using ***x*** *[′]* *−* *i* = ***x*** *−* *i*

Taking the expectation of eq. (A.7),

**E** *x* *i* *′* *[∼]* *[p]* [(] *[·|]* ***[x]*** *[−]* *[i]* [)] � *f* ( ***x*** *[′]* ) � *≤* *f* ( ***x*** ) + log *p* ( *x* *i* *|* ***x*** *−* *i* ) *−* **E** *x* *i* *′* *[∼]* *[p]* [(] *[·|]* ***[x]*** *[−]* *[i]* [)] �log *p* ( *x* *i* *[′]* *[|]* ***[ x]*** *[−]* *[i]* [)] �

= *f* ( ***x*** ) *−* S [ *p* ( *x* *i* *|* ***x*** *−* *i* )] + H [ *p* ( *· |* ***x*** *−* *i* )] .

That is, the energy is expected to decrease if the expected surprise of
the new sample *x* *i* *[′]* *[|]* ***[ x]*** *[−]* *[i]* [ is smaller than the surprise of the current]
sample *x* *i* *|* ***x*** *−* *i* .

**Solution to exercise 6.30 (Hamiltonian Monte Carlo).**

1. We show that under the dynamics (6.49), the Hamiltonian *H* ( ***x***, ***y*** )
is a constant. In particular, *H* ( ***x*** *[′]*, ***y*** *[′]* ) = *H* ( ***x***, ***y*** ) . This directly implies that

*α* ( ***x*** *[′]* *|* ***x*** ) = min *{* 1, exp ( *H* ( ***x*** *[′]*, ***y*** *[′]* ) *−* *H* ( ***x***, ***y*** )) *}* = 1.

To see why *H* ( ***x***, ***y*** ) is constant, we compute


*d* *[d]* ***[x]***
*dt* *[H]* [(] ***[x]*** [,] ***[ y]*** [) =] ***[ ∇]*** ***[x]*** *[ H]* *[ ·]* *dt*



*[d]* ***[x]*** *[d]* ***[y]***

*dt* [+] ***[ ∇]*** ***[y]*** *[ H]* *[ ·]* *dt*


using the chain rule
*dt*


= ***∇*** ***x*** *H* *·* ***∇*** ***y*** *H* *−* ***∇*** ***x*** *H* *·* ***∇*** ***y*** *H* using the Hamiltonian dynamics (6.49)

= 0.

2. By applying one Leapfrog step, we have for ***x***,

***x*** ( *t* + *τ* ) = ***x*** ( *t* ) + *τ* ***y*** ( *t* + *[τ]* / 2 ) using eq. (6.50b)

solutions 297


= ***x*** ( *t* ) + *τ* ***y*** ( *t* ) *−* *[τ]* using eq. (6.50a)
� 2 ***[∇]*** ***[x]*** *[ f]* [ (] ***[x]*** [(] *[t]* [))] �


= ***x*** ( *t* ) *−* *[τ]* 2 [2] ***[∇]*** ***[x]*** *[ f]* [ (] ***[x]*** [(] *[t]* [)) +] *[ τ]* ***[y]*** [(] *[t]* [)] [.]


Now observe that ***y*** ( *t* ) is a Gaussian random variable, independent of ***x*** (because we sample ***y*** freshly at the beginning of the
*L* Leapfrog steps, and we are doing just one Leapfrog step). By

. 2 .
renaming *τ* *[′]* = *τ* / 2 and *ϵ* = ***y*** ( *t* ), we get


***x*** *t* + 1 = ***x*** *t* *−* *τ* *[′]* ***∇*** ***x*** *f* ( ***x*** *t* ) + *√* 2 *τ* *[′]* *ϵ*


which coincides with the proposal distribution of Langevin Monte
Carlo (6.44).

*Bayesian Deep Learning*

**Solution to exercise 7.2 (Softmax is a generalization of the logistic**
**function).** We have

ex p ( *f* 1 )
*σ* 1 ( ***f*** ) = using the definition of the softmax
exp ( *f* 0 ) + exp ( *f* 1 ) function (7.5)

1
= exp ( *f* 0 *−* *f* 1 ) + 1 multiplying by exp [ex] [p] ( [(] *−* *[−]* *f* *[f]* 1 [1] ) [)]

= *σ* ( *−* ( *f* 0 *−* *f* 1 )) . using the definition of the logistic
function (5.10)
*σ* 0 ( ***f*** ) = 1 *−* *σ* ( *f* ) follows from the fact that *σ* 0 ( ***f*** ) + *σ* 1 ( ***f*** ) = 1.

*Active Learning*

**Solution to exercise 8.5 (Mutual information and KL divergence).**

We have

I ( **X** ; **Y** ) = H [ **X** ] *−* H [ **X** *|* **Y** ] using the definition of mutual
information (8.9)
= **E** ***x*** [ *−* log *p* ( ***x*** )] *−* **E** ( ***x***, ***y*** ) [ *−* log *p* ( ***x*** *|* ***y*** )] using the definitions of entropy (5.29)
and conditional entropy (8.2)
= **E** ( ***x***, ***y*** ) [ *−* log *p* ( ***x*** )] *−* **E** ( ***x***, ***y*** ) [ *−* log *p* ( ***x*** *|* ***y*** )] using the law of total expectation (1.15)



*[|]* ***[y]*** [)]

= **E** ( ***x***, ***y*** ) �log *[p]* [(] *p* ***[x]*** ( ***x*** )


.
�


From this we get directly that

I ( **X** ; **Y** ) = **E** ***y*** **E** ***x*** *|* ***y***


log *[p]* [(] ***[x]*** *[|]* ***[y]*** [)]
� *p* ( ***x*** )


log *[p]* [(] ***[x]*** *[|]* ***[y]*** [)]
� *p* ( ***x*** )


�


= **E** ***y*** [ KL ( *p* ( ***x*** *|* ***y*** ) *∥* *p* ( ***x*** ))], using the definition of KL-divergence
(5.43)
and we also conclude


using the definition of conditional
probability (1.11)


I ( **X** ; **Y** ) = **E** ( ***x***, ***y*** )


*p* ( ***x***, ***y*** )
log
� *p* ( ***x*** ) *p* ( ***y*** ) �

298 probabilistic artificial intelligence

= KL ( *p* ( ***x***, ***y*** ) *∥* *p* ( ***x*** ) *p* ( ***y*** )) . using the definition of KL-divergence
(5.43)

**Solution to exercise 8.8 (Non-monotonicity of cond. mutual informa-**
**tion).** Symmetry of conditional mutual information (8.18) and eq. (8.17)
give the following relationship,

I ( **X** ; **Y**, **Z** ) = I ( **X** ; **Y** ) + I ( **X** ; **Z** *|* **Y** ) = I ( **X** ; **Z** ) + I ( **X** ; **Y** *|* **Z** ) . (A.8)

1. **X** *⊥* **Z** implies I ( **X** ; **Z** ) = 0. Thus, eq. (A.8) simplifies to

I ( **X** ; **Y** ) + I ( **X** ; **Z** *|* **Y** ) = I ( **X** ; **Y** *|* **Z** ) .

Using that I ( **X** ; **Z** *|* **Y** ) *≥* 0, we conclude I ( **X** ; **Y** *|* **Z** ) *≥* I ( **X** ; **Y** ) .
2. **X** *⊥* **Z** *|* **Y** implies I ( **X** ; **Z** *|* **Y** ) = 0. Equation (A.8) simplifies to

I ( **X** ; **Y** ) = I ( **X** ; **Z** ) + I ( **X** ; **Y** *|* **Z** ) .

Using that I ( **X** ; **Z** ) *≥* 0, we conclude I ( **X** ; **Y** *|* **Z** ) *≤* I ( **X** ; **Y** ) .
3. Again, eq. (A.8) simplifies to

I ( **X** ; **Y** ) = I ( **X** ; **Z** ) + I ( **X** ; **Y** *|* **Z** ) .

Using that I ( **X** ; **Y** *|* **Z** ) *≥* 0, we conclude I ( **X** ; **Z** ) *≤* I ( **X** ; **Y** ) .
4. Expanding the definition of interaction information, one obtains

I ( **X** ; **Y** ; **Z** ) = ( H [ **X** ] + H [ **Y** ] + H [ **Z** ])

*−* ( H [ **X**, **Y** ] + H [ **X**, **Z** ] + H [ **Y**, **Z** ])

+ H [ **X**, **Y**, **Z** ],

and hence, interaction information is symmetric.
5. Conditional on either one of *X* 1 or *X* 2, the distribution of *Y* remains unchanged, and hence I ( *Y* ; *X* 1 *|* *X* 2 ) = I ( *Y* ; *X* 2 *|* *X* 1 ) = 0.
Conversely, conditional on both *X* 1 and *X* 2, *Y* is fully determined,
and hence I ( *Y* ; *X* 1, *X* 2 ) = 1 noting that *Y* encodes one bit worth of
information. Thus, I ( *Y* ; *X* 1 ; *X* 2 ) = *−* 1 meaning that there is synergy between *X* 1 and *X* 2 with respect to *Y* .

**Solution to exercise 8.10 (Marginal gain of maximizing mutual in-**
**formation).** As suggested, we derive the result in two steps.

1. First, we have

∆ *I* ( ***x*** *|* *A* ) = *I* ( *A* *∪{* ***x*** *}* ) *−* *I* ( *A* ) using the definition of marginal gain
(8.23)
= I ( ***f*** *A* *∪{* ***x*** *}* ; ***y*** *A*, *y* ***x*** ) *−* I ( ***f*** *A* ; ***y*** *A* ) using the definition of *I* (8.22)

= I ( ***f*** *A* *∪{* ***x*** *}* ; ***y*** *A*, *y* ***x*** ) *−* I ( ***f*** *A* *∪{* ***x*** *}* ; ***y*** *A* ) using ***y*** *A* *⊥* *f* ***x*** *|* ***f*** *A*

= H [ ***f*** *A* *∪{* ***x*** *}* *|* ***y*** *A* ] *−* H [ ***f*** *A* *∪{* ***x*** *}* *|* ***y*** *A*, *y* ***x*** ] using the definition of MI (8.9)

= I ( ***f*** *A* *∪{* ***x*** *}* ; *y* ***x*** *|* ***y*** *A* ) using the definition of cond. MI (8.15)

= I ( ***f*** *A* ; *y* ***x*** *|* *f* ***x***, ***y*** *A* ) + I ( *f* ***x*** ; *y* ***x*** *|* ***y*** *A* ) using eq. (8.17)

= I ( *f* ***x*** ; *y* ***x*** *|* ***y*** *A* ) . using I ( ***f*** *A* ; *y* ***x*** *|* *f* ***x***, ***y*** *A* ) = 0 as
*y* ***x*** *⊥* ***f*** *A* *|* *f* ***x***

solutions 299

2. For the second part, we get

I ( *f* ***x*** ; *y* ***x*** *|* ***y*** *A* ) = I ( *y* ***x*** ; *f* ***x*** *|* ***y*** *A* ) using symmetry of conditional MI (8.18)

= H [ *y* ***x*** *|* ***y*** *A* ] *−* H [ *y* ***x*** *|* *f* ***x***, ***y*** *A* ] using the definition of cond. MI (8.15)

= H [ *y* ***x*** *|* ***y*** *A* ] *−* H [ *y* ***x*** *|* *f* ***x*** ] using that *y* ***x*** *⊥* ***ϵ*** *A* so *y* ***x*** *⊥* ***y*** *A* *|* *f* ***x***

= H [ *y* ***x*** *|* ***y*** *A* ] *−* H [ *ϵ* ***x*** ] . given *f* ***x***, the only randomness in *y* ***x***
originates from *ϵ* ***x***

*Bayesian Optimization*

**Solution to exercise 9.2 (Convergence to the static optimum).** First,

observe that


lim *R* *T* *f* *[⋆]* ( ***x*** ) *−* lim 1
*T* *→* ∞ *T* [=] [ max] ***x*** *T* *→* ∞ *T*


*T*
###### ∑ f [⋆] ( x t ) using the definition of regret (9.4)

*t* = 1


= max *f* *[⋆]* ( ***x*** ) *−* lim (A.9) using the Cesàro mean (9.7)
***x*** *t* *→* ∞ *[f]* *[ ⋆]* [(] ***[x]*** *[t]* [)] [.]

Note that existence of lim *t* *→* ∞ ***x*** *t* implies the existence of lim *t* *→* ∞ *f* *[⋆]* ( ***x*** *t* ) .

Now, suppose that the algorithm converges to the static optimum,

lim *f* *[⋆]* ( ***x*** ) .
*t* *→* ∞ *[f]* *[ ⋆]* [(] ***[x]*** *[t]* [) =] [ max] ***x***

Together with eq. (A.9) we conclude that the algorithm achieves sublinear regret.

For the other direction, we prove the contrapositive. That is, we assume that the algorithm does not converge to the static optimum and
show that it has (super-)linear regret. Formally, we assume

lim *f* *[⋆]* ( ***x*** ) .
*t* *→* ∞ *[f]* *[ ⋆]* [(] ***[x]*** *[t]* [)] *[ <]* [ max] ***x***

Together with eq. (A.9) we conclude lim *T* *→* ∞ *[R]* *T* / *T* *>* 0.

**Solution to exercise 9.5 (Bayesian regret for GP-UCB).**

1. We denote the static optimum by ***x*** *[⋆]* . By the definition of ***x*** *t*,

*µ* *t* *−* 1 ( ***x*** *t* ) + *β* *t* *σ* *t* *−* 1 ( ***x*** *t* ) *≥* *µ* *t* *−* 1 ( ***x*** *[⋆]* ) + *β* *t* *σ* *t* *−* 1 ( ***x*** *[⋆]* )

*≥* *f* *[⋆]* ( ***x*** *[⋆]* ) . using eq. (9.11)

Thus,

*r* *t* = *f* *[⋆]* ( ***x*** *[⋆]* ) *−* *f* *[⋆]* ( ***x*** *t* )

*≤* *β* *t* *σ* *t* *−* 1 ( ***x*** *t* ) + *µ* *t* *−* 1 ( ***x*** *t* ) *−* *f* *[⋆]* ( ***x*** *t* )

*≤* 2 *β* *t* *σ* *t* *−* 1 ( ***x*** *t* ) . again using eq. (9.11)

2. We have for any fixed *T*,

I ( ***f*** *T* + 1 ; ***y*** *T* + 1 ) = H [ ***y*** *T* + 1 ] *−* H [ ***ϵ*** *T* + 1 ] analogously to eq. (8.25)

300 probabilistic artificial intelligence

= H [ ***y*** *T* ] *−* H [ ***ϵ*** *T* ] + H� *y* ***x*** *T* + 1 *|* ***y*** *T* � *−* H� *ϵ* ***x*** *T* + 1 � using the chain rule for entropy (8.5)
and the mutual independence of ***ϵ*** *T* + 1
= I ( ***f*** *T* ; ***y*** *T* ) + I� *f* ***x*** *T* + 1 ; *y* ***x*** *T* + 1 *|* ***y*** *T* � using the definition of MI (8.9)


�


= I ( ***f*** *T* ; ***y*** 1: *n* ) + [1]

2 [log]


= I ( ***f*** *T* ; ***y*** 1: *n* ) + [1]


�


1 + *[σ]* *T* [2] [(] ***[x]*** *[T]* [+] [1] [)]

*σ* *n* [2]


. using eq. (8.14)


Note that I ( ***f*** 0 ; ***y*** 0 ) = 0. The result then follows by induction.
3. It follows from hint (b) that it suffices to show ∑ *t* *[T]* = 1 *[r]* *t* [2] *[≤O]* � *β* [2] *T* *[γ]* *[T]* �.
We have


*T*
###### ∑ r t [2] [≤] [4] [β] [2] T

*t* = 1


*T*
###### ∑ σ t [2] − 1 [(] [x] [t] [)] using part (1)

*t* = 1


*σ* *t* [2] *−* 1 [(] ***[x]*** *[t]* [)]

.
*σ* *n* [2]


= 4 *σ* *n* [2] *[β]* [2] *T*


*T*
###### ∑

*t* = 1


Observe that *σ* *t* [2] *−* 1 [(] ***[x]*** *[t]* [)] [/] *[σ]* *n* [2] [is bounded by] *[ M]* = . max ***x*** *∈X* *σ* 02 [(] ***[x]*** [)] [/] *[σ]* *n* [2] [as]
variance is monotonically decreasing (cf. section 1.6). Applying
hint (a), we obtain


�


�


*t* *−* 1 [(] ***[x]*** *[t]* [)]
1 + *[σ]* [2]

*σ* *n* [2]


*t* *−* 1 [(] ***[x]*** *[t]* [)]
1 + *[σ]* [2]


*≤* 4 *Cσ* *n* [2] *[β]* [2] *T*


*T*
###### ∑ log

*t* = 1


= 8 *Cσ* *n* [2] *[β]* [2] *T* [I] [(] ***[ f]*** *[T]* [;] ***[ y]*** *[T]* [)] using part (2)

*≤* 8 *Cσ* *n* [2] *[β]* [2] *T* *[γ]* *[T]* [.] using the definition of *γ* *T* (9.10)

**Solution to exercise 9.7 (Sublinear regret of GP-UCB for a linear**
**kernel).**

1. Let *S* *⊆X* be such that *|* *S* *| ≤* *T* . Recall from eq. (8.14) that
I ( ***f*** *S* ; ***y*** *S* ) = [1] 2 [log det] � ***I*** + *σ* *n* *[−]* [2] ***[K]*** *SS* �. Using that the kernel is linear

we can rewrite ***K*** *SS* = ***X*** *S* *[⊤]* ***[X]*** *[S]* [. Using Weinstein-Aronszajn’s identity]
(9.17) we have


I ( ***f*** *S* ; ***y*** *S* ) = [1]


***I*** + *σ* *n* *[−]* [2] ***[X]*** *S* ***[X]*** *S* *[⊤]* .
2 [log det] � �



[1] ***I*** + *σ* *n* *[−]* [2] ***[X]*** *S* *[⊤]* ***[X]*** *[S]* = [1]

2 [log det] � �


If we define ***M*** = . ***I*** + *σ* *n* *−* 2 ***[X]*** *S* ***[X]*** *S* *[⊤]* [as a sum of symmetric positive]
definite matrices, ***M*** itself is symmetric positive definite. Thus, we
have from Hadamard’s inequality (9.16),

det ( ***M*** ) *≤* det diag *{* ***I*** + *σ* *n* *[−]* [2] ***[X]*** *S* ***[X]*** *S* *[⊤]* *[}]* diag *{* ***A*** *}* refers to the diagonal matrix
� �
whose elements are those of ***A***
= det ***I*** + *σ* *n* *[−]* [2] [diag] *[{]* ***[X]*** *S* ***[X]*** *S* *[⊤]* *[}]* .
� �

Note that

*|* *S* *|*
###### diag { X S X S [⊤] [}] [(] [i] [,] [ i] [) =] ∑ x t ( i ) [2]

*t* = 1


*d* *|* *S* *|* *|* *S* *|*
###### ≤ ∑ ∑ x t ( i ) [2] = ∑ ∥ x t ∥ [2] 2

*i* = 1 *t* = 1 *t* = 1 ����

*≤* 1


*≤|* *S* *| ≤* *T* .

solutions 301


If we denote by *λ* *≤* *T* the largest term of diag *{* ***X*** *S* ***X*** *S* *[⊤]* *[}]* [ then we]
have

det ( ***M*** ) *≤* ( 1 + *σ* *n* *[−]* [2] *[λ]* [)] *[d]* *[ ≤]* [(] [1] [ +] *[ σ]* *n* *[−]* [2] *[T]* [)] *[d]* [,]

yielding,

I ( ***f*** *S* ; ***y*** *S* ) *≤* *[d]* *n* *[T]* [)]

2 [log] [(] [1] [ +] *[ σ]* *[−]* [2]

implying that *γ* *T* = *O* ( *d* log *T* ) .
2. Using the Bayesian regret bound (cf. theorem 9.4) and then *γ* *T* =
*O* ( *d* log *T* ), we have


*dT*,
�


*R* *T* = *O* [˜]
��


*γ* *T* *T* = *O* [˜] *√*
� �


*R* *T*
and hence, lim *T* *→* ∞ *T* [=] [ 0.]

**Solution to exercise 9.9 (Closed-form expected improvement).**

1. Note that *f* is a Gaussian process, and hence, our posterior distribution after round *t* *−* 1 is entirely defined by the mean function
*µ* *t* *−* 1 and the covariance function *k* *t* *−* 1 . Reparameterizing the posterior distribution using a standard Gaussian (1.122), we obtain

*f* *t* *−* 1 ( ***x*** ) = *µ* *t* *−* 1 ( ***x*** ) + *σ* *t* *−* 1 ( ***x*** ) *ϵ*

for *ϵ* *∼N* ( 0, 1 ) . We get

EI *t* ( ***x*** ) = **E** *f* *t* *−* 1 ( ***x*** ) *∼N* ( *µ* *t* *−* 1 ( ***x*** ), *σ* *t* 2 *−* 1 [(] ***[x]*** [))] � ( *f* *t* *−* 1 ( ***x*** ) *−* *f* [ˆ] ) + � using the definition of expected
improvement (9.20)

= **E** *ϵ* *∼N* ( 0,1 ) � ( *µ* *t* *−* 1 ( ***x*** ) + *σ* *t* *−* 1 ( ***x*** ) *ϵ* *−* *f* [ˆ] ) + � using the reparameterization

+ ∞

=
� *−* ∞ [(] *[µ]* *[t]* *[−]* [1] [(] ***[x]*** [) +] *[ σ]* *[t]* *[−]* [1] [(] ***[x]*** [)] *[ϵ]* *[ −]* *[f]* [ˆ] [ )] [+] *[ ·]* *[ ϕ]* [(] *[ϵ]* [)] *[ d][ϵ]* [.]

ˆ
*f* *−* *µ* *t* *−* 1 ( ***x*** )
For *ϵ* *<* *σ* *t* *−* 1 ( ***x*** ) we have ( *µ* *t* *−* 1 ( ***x*** ) + *σ* *t* *−* 1 ( ***x*** ) *ϵ* *−* *f* [ˆ] ) + = 0. Thus,

we obtain

+ ∞
EI *t* ( ***x*** ) =
� ( *f* [ˆ] *−* *µt* *−* 1 ( ***x*** )) / *σt* *−* 1 ( ***x*** ) [(] *[µ]* *[t]* *[−]* [1] [(] ***[x]*** [) +] *[ σ]* *[t]* *[−]* [1] [(] ***[x]*** [)] *[ϵ]* *[ −]* *[f]* [ˆ] [ )] *[ ·]* *[ ϕ]* [(] *[ϵ]* [)] *[ d][ϵ]* [. (A.][10][)]

2. By splitting the integral from eq. (A.10) into two distinct terms, we

obtain

+ ∞
EI *t* ( ***x*** ) = ( *µ* *t* *−* 1 ( ***x*** ) *−* *f* [ˆ] )
� ( *f* [ˆ] *−* *µt* *−* 1 ( ***x*** )) / *σt* *−* 1 ( ***x*** ) *[ϕ]* [(] *[ϵ]* [)] *[ d][ϵ]*

+ ∞
*−* *σ* *t* *−* 1 ( ***x*** )
� ( *f* [ˆ] *−* *µt* *−* 1 ( ***x*** )) / *σt* *−* 1 ( ***x*** ) [(] *[−]* *[ϵ]* [)] *[ ·]* *[ ϕ]* [(] *[ϵ]* [)] *[ d][ϵ]* [.]

302 probabilistic artificial intelligence

For the first term, we use the symmetry of *N* ( 0, 1 ) around 0 to
write the integral in terms of the CDF. For the second term, we
notice that ( *−* *ϵ* ) *·* *ϕ* ( *ϵ* ) = *√* 1 2 *π* *d* *d* *ϵ* *[e]* *[−]* *[ϵ]* [2] [/] [2] [. Thus, we can derive this]

integral directly,


�


= ( *µ* *t* *−* 1 ( ***x*** ) *−* *f* [ˆ] ) Φ


*µ* *t* *−* 1 ( ***x*** ) *−* *f* [ˆ]

*σ* *t* *−* 1 ( ***x*** )

�


.

��


*−* *σ* *t* *−* 1 ( ***x*** )


�


lim
*ϵ* *→* ∞ *[ϕ]* [(] *[ϵ]* [)] *[ −]* *[ϕ]*


ˆ
*f* *−* *µ* *t* *−* 1 ( ***x*** )

*σ* *t* *−* 1 ( ***x*** )

�


Using the symmetry of *ϕ* around 0, we obtain


�


.

�


EI *t* ( ***x*** ) = ( *µ* *t* *−* 1 ( ***x*** ) *−* *f* [ˆ] ) Φ


*µ* *t* *−* 1 ( ***x*** ) *−* *f* [ˆ]

*σ* *t* *−* 1 ( ***x*** )

�


+ *σ* *t* *−* 1 ( ***x*** ) *ϕ*


*µ* *t* *−* 1 ( ***x*** ) *−* *f* [ˆ]

*σ* *t* *−* 1 ( ***x*** )

�


*Markov Decision Processes*

**Solution to exercise 10.8 (Value functions).** We can use eq. (10.15)
to write the state-action values as a linear system of equations (i.e.,
as a “table”). This linear system can be solved, for example, using
Gaussian elimination to yield the desired result.

**Solution to exercise 10.13 (Greedy policies).** It follows directly from
the definition of the state-action value function (10.9) that
###### arg max q ( x, a ) = arg max r ( x, a ) + γ ∑ p ( x [′] | x, a ) · v ( x [′] ) .

*a* *∈* *A* *a* *∈* *A* *x* *[′]* *∈* *X*

**Solution to exercise 10.16 (Optimal policies).**

1. Recall from Bellman’s theorem (10.32) that a policy is optimal iff it
is greedy with respect to its state-action value function. Now, observe that in the “poor, unknown” state, the policy *π* is not greedy.
2. Analogously to exercise 10.8, we write the state-action values as a
linear system of equations and solve the system using, e.g., Gaus
sian elimination.

3. Observe from the result of (2) that *π* *[′]* is greedy with respect to its
state-action value function, and hence, it follows from Bellman’s
theorem that *π* *[′]* is optimal.

**Solution to exercise 10.23 (Linear convergence of policy iteration).**
Using the hint and ***v*** *[⋆]* *≥* ***v*** *[π]* for any policy *π*,

*∥* ***v*** *[π]* *[t]* *−* ***v*** *[⋆]* *∥* ∞ *≤∥* ***B*** *[⋆]* ***v*** *[π]* *[t]* *[−]* [1] *−* ***v*** *[⋆]* *∥* ∞

= *∥* ***B*** *[⋆]* ***v*** *[π]* *[t]* *[−]* [1] *−* ***B*** *[⋆]* ***v*** *[⋆]* *∥* ∞ using that ***v*** *[⋆]* is a fixed-point of ***B*** *[⋆]*, that
is, ***B*** *[⋆]* ***v*** *[⋆]* = ***v*** *[⋆]*
*≤* *γ* *∥* ***v*** *[π]* *[t]* *[−]* [1] *−* ***v*** *[⋆]* *∥* ∞ using that ***B*** *[⋆]* is a contraction, see
theorem 10.21
*≤* *γ* *[t]* *∥* ***v*** *[π]* [0] *−* ***v*** *[⋆]* *∥* ∞ . by induction

solutions 303

**Solution to exercise 10.24 (Reward modification).**

1. Recall that the value function *v* *[π]* *M* [for an MDP] *[ M]* [ is defined as]
*v* *[π]* *M* [(] *[x]* [) =] **[ E]** *[π]* �∑ [∞] *t* = 0 *[γ]* *[t]* *[R]* *[t]* �� *X* 0 = *x* �. Given an optimal policy *π* *[⋆]* for
*M* and any policy *π*, we know that for any *x* *∈* *X*,

*v* *[π]* *M* *[⋆]* [(] *[x]* [)] *[ ≥]* *[v]* *[π]* *M* [(] *[x]* [)]

*⇐⇒* **E** *π* *⋆* [�] ∑ [∞] *t* = 0 *[γ]* *[t]* *[R]* *[t]* �� *X* 0 = *x* � *≥* **E** *π* �∑ [∞] *t* = 0 *[γ]* *[t]* *[R]* *[t]* �� *X* 0 = *x* �

*⇐⇒* **E** *π* *⋆* [�] ∑ [∞] *t* = 0 *[γ]* *[t]* *[α][R]* *[t]* �� *X* 0 = *x* � *≥* **E** *π* �∑ [∞] *t* = 0 *[γ]* *[t]* *[α][R]* *[t]* �� *X* 0 = *x* � multiplying both sides by *α*

*⇐⇒* **E** *π* *⋆* [�] ∑ [∞] *t* = 0 *[γ]* *[t]* *[R]* *[′]* *t* �� *X* 0 = *x* � *≥* **E** *π* �∑ [∞] *t* = 0 *[γ]* *[t]* *[R]* *[′]* *t* �� *X* 0 = *x* �

*⇐⇒* *v* *[π]* *[⋆]*
*M* *[′]* [(] *[x]* [)] *[ ≥]* *[v]* *[π]* *M* *[′]* [(] *[x]* [)] [.]

Thus, *π* *[⋆]* is an optimal policy for *M* *[′]* .
2. We give an example where the optimal policies differ when re
wards are shifted.

Consider an MDP with three states *{* 1, 2, 3 *}* where 1 is the initial
state and 3 is a terminal state. If one plays action *A* in states 1
or 2 one transitions directly to the terminal state. Additionally, in
state 1 one can play action *B* which leads to state 2. Let every
transition give a deterministic reward of *r* = . *−* 1. Then it is optimal
to traverse the shortest path to the terminal state, in particular, to

choose action *A* when in state 1.

If we consider the reward *r* *[′]* = . *r* + 2 = 1, then it is optimal to traverse the longest path to the terminal state, in particular, to choose

action *B* when in state 1.

3. For an MDP *M*, we know that is optimal state-action value function
satisfies Bellman’s optimality equation (10.35),


*q* *[⋆]* *M* [(] *[x]* [,] *[ a]* [) =] **[ E]** *x* *[′]* *|* *x*, *a*

For the MDP *M* *[′]*, we have


� *r* ( *x*, *x* *[′]* ) + *γ* max *a* *[′]* *∈* *A* *[q]* *[⋆]* *M* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] �.


*q* *[⋆]* *M* *[′]* [(] *[x]* [,] *[ a]* [) =] **[ E]** *x* *[′]* *|* *x*, *a*

= **E** *x* *′* *|* *x*, *a*

= **E** *x* *′* *|* *x*, *a*


� *r* *[′]* ( *x*, *x* *[′]* ) + *γ* max *a* *[′]* *∈* *A* *[q]* *[⋆]* *M* *[′]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] �

� *r* ( *x*, *x* *[′]* ) + *f* ( *x*, *x* *[′]* ) + *γ* max *a* *[′]* *∈* *A* *[q]* *[⋆]* *M* *[′]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] �

� *r* ( *x*, *x* *[′]* ) + *γϕ* ( *x* *[′]* ) *−* *ϕ* ( *x* ) + *γ* max *a* *[′]* *∈* *A* *[q]* *[⋆]* *M* *[′]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] �.


Reorganizing the terms, we obtain


� *q* *[⋆]* *M* *[′]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [) +] *[ ϕ]* [(] *[x]* *[′]* [)] � [�] .


*q* *[⋆]* *M* *[′]* [(] *[x]* [,] *[ a]* [) +] *[ ϕ]* [(] *[x]* [) =] **[ E]** *x* *[′]* *|* *x*, *a*


*r* ( *x*, *x* *[′]* ) + *γ* max
� *a* *[′]* *∈* *A*


If we now define *q* ( *x*, *a* ) = . *q* *⋆* *M* *[′]* [(] *[x]* [,] *[ a]* [) +] *[ ϕ]* [(] *[x]* [)] [, we have]


*q* ( *x*, *a* ) = **E** *x* *′* *|* *x*, *a*


� *r* ( *x*, *x* *[′]* ) + *γ* max *a* *[′]* *∈* *A* *[q]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [)] �.

304 probabilistic artificial intelligence

This is exactly Bellman’s optimality equation for the MDP *M* with
reward function *r*, and hence, *q* *≡* *q* *[⋆]* *M* [.]
If we take *π* *[⋆]* to be an optimal policy for *M*, then it satisfies

*π* *[⋆]* ( *x* ) *∈* arg max *q* *[⋆]* *M* [(] *[x]* [,] *[ a]* [)]
*a* *∈* *A*

= arg max *q* *[⋆]* *M* *[′]* [(] *[x]* [,] *[ a]* [) +] *[ ϕ]* [(] *[x]* [)] using the above characterization of *q* *[⋆]* *M*
*a* *∈* *A*

= arg max *q* *[⋆]* *M* *[′]* [(] *[x]* [,] *[ a]* [)] [.] using that *ϕ* ( *x* ) is independent of *a*
*a* *∈* *A*

*Tabular Reinforcement Learning*

**Solution to exercise 11.15 (Q-learning).**

1. *Q* *[⋆]* ( *A*, *↓* ) = 1.355, *Q* *[⋆]* ( *G* 1, exit ) = 5.345, *Q* *[⋆]* ( *G* 2, exit ) = 0.5
2. Repeating the given episodes infinitely often will not lead to convergence to the optimal Q-function because not all state-action
pairs are visited infinitely often.
Let us assume we observe the following episode instead of the first
episode.

E p isode 3
*x* *a* *x* *[′]* *r*

*A* *→* *B* 0

*B* *↓* *G* 2 0

*G* 2 exit 1

If we repeat episodes 2 and 3 infinitely often, Q-learning will converge to the optimal Q-function as all state-action pairs will be
visited infinitely often.
3. First, recall that Q-learning is an off-policy algorithm, and hence,
even if episodes are obtained off-policy, Q-learning will still converge to the optimal Q-function (if the other convergence conditions are met). Note that it only matters which state-action pairs
are observed and not which policies were followed to obtain these

observations.

The “closer” the initial Q-values are to the optimal Q-function,
the faster the convergence of Q-learning. However, if the convergence conditions are met, Q-learning will converge to the optimal
Q-function regardless of the initial Q-values.
4. *v* *[⋆]* ( *A* ) = 10, *v* *[⋆]* ( *B* ) = 10, *v* *[⋆]* ( *G* 1 ) = 10, *v* *[⋆]* ( *G* 2 ) = 1

solutions 305


*Model-free Approximate Reinforcement Learning*

**Solution to exercise 12.2 (Q-learning and function approximation).**

1. We have to show that

*v* *[⋆]* ( *x* ) = max � *v* *[⋆]* ( *x* *[′]* ) �
*a* *∈* *A* *[r]* [(] *[x]* [,] *[ a]* [) +] *[ γ]* **[E]** *[x]* *[′]* *[|]* *[x]* [,] *[a]*

for every *x* *∈{* 1, 2, . . ., 7 *}* . We give a derivation here for *x* = 1 and

*x* = 2.

  - For *x* = 1,

*v* *[⋆]* ( 1 ) = *−* 3

max � *v* *[⋆]* ( *x* *[′]* ) � = *−* 3
*a* *∈* *A* *[r]* [(] [1,] *[ a]* [) +] *[ γ]* **[E]** *[x]* *[′]* *[|]* [1,] *[a]*

since


*−* 1 + *−* 3 = *−* 4 if *a* = *−* 1.


*r* ( 1, *a* ) + *γ* **E** *x* *′* *|* 1, *a* � *v* *[⋆]* ( *x* *[′]* ) � =

- Likewise, for *x* = 2,






 *−−* 11 + + *− −* 23 = = *− −* 34 ifif *a a* = = 1 *−*




*v* *[⋆]* ( 2 ) = *−* 2

max � *v* *[⋆]* ( *x* *[′]* ) � = *−* 2
*a* *∈* *A* *[r]* [(] [2,] *[ a]* [) +] *[ γ]* **[E]** *[x]* *[′]* *[|]* [2,] *[a]*


since

*r* ( 2, *a* ) + *γ* **E** *x* *′* *|* 2, *a* � *v* *[⋆]* ( *x* *[′]* ) � =

2. We have






*−* 1 + *−* 3 = *−* 4 if *a* = *−* 1.


 *−−* 11 + + *− −* 13 = = *− −* 24 ifif *a a* = = 1 *−*




*Q* ( 3, *−* 1 ) = 0 + [1]

2


2


*Q* ( 2, 1 ) = 0 + [1]

2


2


*Q* ( 3, 1 ) = 0 + [1]

2


*−* 1 + max = [1]
� *a* *[′]* *∈* *A* *[Q]* [(] [2,] *[ a]* *[′]* [)] � 2

*−* 1 + max = [1]
� *a* *[′]* *∈* *A* *[Q]* [(] [3,] *[ a]* *[′]* [)] � 2

*−* 1 + max = [1]
� *a* *[′]* *∈* *A* *[Q]* [(] [4,] *[ a]* *[′]* [)] � 2



[1]

2 [(] *[−]* [1] [ +] [ 0] [) =] *[ −]* [1] 2

[1]

2 [(] *[−]* [1] [ +] [ 0] [) =] *[ −]* [1] 2

[1]

2 [(] *[−]* [1] [ +] [ 0] [) =] *[ −]* [1] 2


2


*Q* ( 4, 1 ) = 0 + [1]

2


0 + max = [1]
� *a* *[′]* *∈* *A* *[Q]* [(] [4,] *[ a]* *[′]* [)] � 2 [(] [0] [ +] [ 0] [) =] [ 0.]


3. We compute

*x*


***∇*** ***w*** *ℓ* ( ***w*** ; *τ* ) = *−* � *r* + *γ* max *a* *[′]* *∈* *A* *[Q]* [(] *[x]* *[′]* [,] *[ a]* *[′]* [;] ***[ w]*** [old] [)] *[ −]* *[Q]* [(] *[x]* [,] *[ a]* [;] ***[ w]*** [)] � *a*

1





 using the derivation of eq. (12.15)


 2
= *−* *−* 1 + max *−* 1
� *a* *[′]* *∈* *A* *[{]* [1] *[ −]* *[a]* *[′]* *[ −]* [2] *[} −]* [(] *[−]* [2] *[ −]* [1] [ +] [ 1] [)] �

1







306 probabilistic artificial intelligence





=


 *−* 4

2

*−* 2





.



This gives


***w*** *[′]* = ***w*** *−* *α* ***∇*** ***w*** *ℓ* ( ***w*** ; *τ* )




 =





 0

1 /

 3 /


1 / 2

3 / 2





*−* [1]

4






.



=


 *−* 1

1

1



4


 *−* 4

2

*−* 2



**Solution to exercise 12.7 (Variance of score gradients with baselines).**

1. The result follows directly using that

Var [ *f* ( **X** ) *−* *g* ( **X** )] = Var [ *f* ( **X** )] + Var [ *g* ( **X** )] *−* 2Cov [ *f* ( **X** ), *g* ( **X** )] . using eq. (1.48)

2. Denote by *r* ( *τ* ) the discounted rewards attained by trajectory *τ* .
Let *f* ( *τ* ) = . *r* ( *τ* ) ***∇*** ***θ*** Π ***θ*** ( *τ* ) and *g* ( *τ* ) = . *b* ***∇*** ***θ*** Π ***θ*** ( *τ* ) . Recall that
**E** *τ* *∼* Π ***θ*** [ *g* ( *τ* )] = 0, implying that

Var [ *g* ( *τ* )] = Var [ *b* ***∇*** ***θ*** Π ***θ*** ( *τ* )]

= **E** � ( *b* ***∇*** ***θ*** Π ***θ*** ( *τ* )) [2] [�] . using the definition of variance (1.44)

On the other hand,

Cov [ *f* ( *τ* ), *g* ( *τ* )] = **E** [( *f* ( *τ* ) *−* **E** [ *f* ( *τ* )]) *g* ( *τ* )] using the definition of covariance (1.35)


= **E** [ *f* ( *τ* ) *g* ( *τ* )] *−* **E** [ *f* ( *τ* )] **E** [ *g* ( *τ* )]
����
0

= **E** � *b* *·* *r* ( *τ* ) *·* ( ***∇*** ***θ*** Π ***θ*** ( *τ* )) [2] [�] .

Therefore, if *b* [2] *≤* 2 *b* *·* *r* ( ***x***, ***a*** ) for every state ***x*** *∈X* and action
***a*** *∈A*, then the result follows from eq. (12.38).

**Solution to exercise 12.8 (Score gradients with state-dependent base-**
**lines).** First, observe that


using linearity of expectation (1.24)


, using eq. (12.30)

�

(A.11)


**E** *τ* *∼* Π ***θ*** [ *G* 0 ***∇*** ***θ*** log Π ***θ*** ( *τ* )] = **E** *τ* *∼* Π ***θ***

and hence, it suffices to show


*T*
###### ∑ G 0 ∇ θ log π θ ( a t | x t )
� *t* = 0


**E** *τ* *∼* Π ***θ***


*T*
###### ∑ G 0 ∇ θ log π θ ( a t | x t )
� *t* = 0 �


*T*
###### = E τ ∼ Π θ ∑ ( G 0 − b ( τ 0: t − 1 )) ∇ θ log π θ ( a t | x t )
� *t* = 0


.

�

We prove eq. (A.11) with an induction on *T* . The base case ( *T* = 0) is
satisfied trivially. Fixing any *T* and assuming eq. (A.11) holds for *T*,
we have,


solutions 307

using the tower rule (1.33)

using the induction hypothesis


**E** *τ* *∼* Π ***θ***


*T* + 1
###### ∑ ( G 0 − b ( τ 0: t − 1 )) ∇ θ log π θ ( a t | x t )
� *t* = 0 �


�


*τ* 0: *T*
����� ��


= **E** *τ* 0: *T*

= **E** *τ* 0: *T*


**E** *τ* *T* + 1


*T* + 1
###### ∑ ( G 0 − b ( τ 0: t − 1 )) ∇ θ log π θ ( a t | x t )
� *t* = 0


*T*
###### ∑ G 0 ∇ θ log π θ ( a t | x t )
� *t* = 0


.

�


+ **E** *τ* *T* + 1 [( *G* 0 *−* *b* ( *τ* 0: *T* )) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *T* + 1 *|* ***x*** *T* + 1 ) *|* *τ* 0: *T* ]


.


Using the score function trick for the score function ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *t* *|* ***x*** *t* )
analogously to the proof of lemma 12.6, we have,

**E** *τ* *T* + 1 [ *b* ( *τ* 0: *T* ) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *T* + 1 *|* ***x*** *T* + 1 ) *|* *τ* 0: *T* ]

= **E** ***a*** *T* + 1 [ *b* ( *τ* 0: *T* ) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *T* + 1 *|* ***x*** *T* + 1 ) *|* *τ* 0: *T* ]

= *b* ( *τ* 0: *T* ) *π* ***θ*** ( ***a*** *T* + 1 *|* ***x*** *T* + 1 ) ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *T* + 1 *|* ***x*** *T* + 1 ) *d* ***a*** *T* + 1
�

= *b* ( *τ* 0: *T* ) ***∇*** ***θ*** *π* ***θ*** ( ***a*** *T* + 1 *|* ***x*** *T* + 1 ) *d* ***a*** *T* + 1 using the score function trick,
�
= 0. ***∇*** ***θ*** log *π* ***θ*** ( ***a*** *t* *|* ***x*** *t* ) = ***[∇]*** ***θ*** *[π]* ***θ*** [(] ***[a]*** *t* *[|]* ***[x]*** *t* [)] / *π* ***θ*** ( ***a*** *t* *|* ***x*** *t* )

Thus,


�


*τ* 0: *T*
�����


��


= **E** *τ* 0: *T*


**E** *τ* *T* + 1


*T* + 1
###### ∑ G 0 ∇ θ log π θ ( a t | x t )
� *t* = 0


*T* + 1
###### = E τ ∼ Π θ ∑ G 0 ∇ θ log π θ ( a t | x t )
� *t* = 0


�


. using the tower rule (1.33) again


**Solution to exercise 12.10 (Policy gradients with downstream re-**
**turns).** Each trajectory *τ* is described by four transitions,

*τ* = ( *x* 0, *a* 0, *r* 0, *x* 1, *a* 1, *r* 1, *x* 2, *a* 2, *r* 2, *x* 3, *a* 3, *r* 3, *x* 4 ) .

Moreover, we have given

*∂π* *θ* ( 2 *|* *x* )
*π* *θ* ( 2 *|* *x* ) = *θ*, = + 1

*∂θ*

*π* *θ* ( 1 *|* *x* ) = 1 *−* *θ*, *∂π* *θ* ( 1 *|* *x* ) = *−* 1.

*∂θ*

We first compute the downstream returns for the given episode,


*G* 0:3 = *r* 0 + *γr* 1 + *γ* [2] *r* 2 + *γ* [3] *r* [3] = 1 + [1]



[1]
4 *[·]* [ 1] [ +] 8



[1]

2 *[·]* [ 0] [ +] [ 1] 4



[1]

8 *[·]* [ 1] [ =] [ 11] 8


8

308 probabilistic artificial intelligence


*G* 1:3 = *r* 1 + *γr* 2 + *γ* [2] *r* 3 = 0 + [1]



[3]
4 *[·]* [ 1] [ =] 4



[1]

2 *[·]* [ 1] [ +] [ 1] 4


4


*G* 2:3 = *r* 2 + *γr* 3 = 1 + [1]



[1]

2 *[·]* [ 1] [ =] [ 3] 2


2 2

*G* 3:3 = *r* 3 = 1.


Lastly, we can combine them to compute the policy gradient,

3
###### ∇ θ j ( θ ) ≈ ∑ γ [t] G t :3 ∇ θ log π θ ( a t | x t ) using Monte Carlo approximation of

*t* = 0 eq. (12.43) with a single sample


= 1 *·* [11]


8 *[−]* [1] *[ ·]* [ 1] 2


2 *[·]* [ 3] 4


4 [+] [ 1] *[ ·]* [ 1] 4



[ 3]

2 *[−]* [1] *[ ·]* [ 1] 8


4 [.]


4 *[·]* [ 3] 2


8 *[·]* [ 1] [ =] [ 5] 4


**Solution to exercise 12.12 (Eligibility vector).** We have

***∇*** ***θ*** log *π* ***θ*** ( *a* *|* ***x*** ) = ***[∇]*** ***[θ]*** *[π]* ***[θ]*** [(] *[a]* *[|]* ***[ x]*** [)] using the chain rule

*π* ***θ*** ( *a* *|* ***x*** )

= ***ϕ*** ( ***x***, *a* ) *−* [∑] *[b]* *[∈]* *[A]* ***[ ϕ]*** [(] ***[x]*** [,] *[ b]* [)] [ ex] [p] [(] ***[θ]*** *[⊤]* ***[ϕ]*** [(] ***[x]*** [,] *[ b]* [))] using elementary calculus

∑ *b* *∈* *A* exp ( ***θ*** *[⊤]* ***ϕ*** ( ***x***, *b* ))
###### = ϕ ( x, a ) − ∑ π θ ( b | x ) · ϕ ( x, b ) .

*b* *∈* *A*

**Solution to exercise 12.15 (Policy gradient with an exponential fam-**
**ily).**

1. The distribution on actions is given by

*π* ***θ*** ( *a* *|* *x* ) = *σ* ( *f* ***θ*** ( *x* )) *[a]* *·* ( 1 *−* *σ* ( *f* ***θ*** ( *x* ))) [1] *[−]* *[a]* .

To simplify the notation, we write **E** for **E** ( ***x***, *a* ) *∼* *π* ***θ*** and *q* for *q* *[π]* ***[θ]*** . We
get

***∇*** ***θ*** *j* ( ***θ*** )

= **E** [ *q* ( ***x***, *a* ) ***∇*** ***θ*** ( *a* log *σ* ( *f* ***θ*** ( ***x*** )) + ( 1 *−* *a* ) log ( 1 *−* *σ* ( *f* ***θ*** ( ***x*** ))))] using eq. (12.59)

= **E** � *q* ( ***x***, *a* ) ***∇*** *f* ( *a* log *σ* ( *f* ) + ( 1 *−* *a* ) log ( 1 *−* *σ* ( *f* ))) ***∇*** ***θ*** *f* ***θ*** ( ***x*** ) � using the chain rule


��


*e* *[−]* *[f]*

1 + *e* *[−]* *[f]*

�


�


�


*−* *a* log ( 1 + *e* *[−]* *[f]* ) + ( 1 *−* *a* ) log


***∇*** ***θ*** *f* ***θ*** ( ***x*** )


= **E**


�


*q* ( ***x***, *a* ) ***∇*** *f*


= **E** � *q* ( ***x***, *a* ) ***∇*** *f* � *−* *f* + *a f* *−* log ( 1 + *e* *[−]* *[f]* ) � ***∇*** ***θ*** *f* ***θ*** ( ***x*** ) �


�


***∇*** ***θ*** *f* ***θ*** ( ***x*** )

�


�


*e* *[−]* *[f]*
*a* *−* 1 +

1 + *e* *[−]* *[f]*


*e* *[−]* *[f]*
*a* *−* 1 +


= **E**


�


*q* ( ***x***, *a* )


= **E** [ *q* ( ***x***, *a* )( *a* *−* *σ* ( *f* )) ***∇*** ***θ*** *f* ***θ*** ( ***x*** )] .

The term *a* *−* *σ* ( *f* ) can be understood as a residual as it corresponds
to the difference between the target action *a* and the expected action *σ* ( *f* ) .

solutions 309

2. We have

***∇*** ***θ*** *j* ( ***θ*** ) = **E** � *q* ( ***x***, *a* ) ***∇*** *f* log *π* *f* ( *a* *|* ***x*** ) ***∇*** ***θ*** *f* ( ***x*** ) � using eq. (12.59) and the chain rule

= **E** � *q* ( ***x***, *a* ) ***∇*** *f* ( log *h* ( *a* ) + *a f* *−* *A* ( *f* )) ***∇*** ***θ*** *f* ( ***x*** ) �

= **E** � *q* ( ***x***, *a* )( *a* *−* ***∇*** *f* *A* ( *f* )) ***∇*** ***θ*** *f* ( ***x*** ) �.

3. We have ***∇*** *f* *A* ( *f* ) = *σ* ( *f* ) . We are therefore looking for a func
tion *A* ( *f* ) whose derivative is *σ* ( *f* ) = 1 + 1 *e* *[−]* *[f]* [ =] 1 + *e* *[f]* *e* *[f]* [ .] With this

equality of the sigmoid we can find the integral, and we have
*A* ( *f* ) = log ( 1 + *e* *[f]* ) + *c* . Let us confirm that this gives us the Ber
noulli distribution with *c* = 0:

*π* *f* ( *a* *|* ***x*** ) = *h* ( *a* ) exp ( *a f* *−* log ( 1 + *e* *[f]* ))

*e* *[a f]*
= *h* ( *a* )

1 + *e* *[f]*


1 *−* *σ* ( *f* ) if *a* = 0.


= *h* ( *a* )






*σ* ( *f* ) if *a* = 1

1 *−* *σ* ( *f* ) if *a* = 0.




This is the Bernoulli distribution with parameter *σ* ( *f* ) where we
have *h* ( *a* ) = 1.
4. Using that ***∇*** *f* *A* ( *f* ) = *f*, we immediately get

***∇*** ***θ*** *j* ( ***θ*** ) = **E** [ *q* ( ***x***, *a* )( *a* *−* *f* ) ***∇*** ***θ*** *f* ( ***x*** )] .

5. No, we cannot use the reparameterization trick since we do not
know how the states ***x*** depend on action *a* . These dependencies
are determined by the the unknown dynamics of the environment.
Nonetheless, we can apply it after sampling an episode according
to a policy and then updating policy parameters in hindsight. This
is for example done by the soft actor-critic (SAC) algorithm (cf.
section 12.4.6).

### *Bibliography*

Eitan Altman. *Constrained Markov decision processes: stochastic modeling* .
Routledge, 1999.

Yarden As, Ilnura Usmanova, Sebastian Curi, and Andreas Krause.

Constrained policy optimization via bayesian world models. *arXiv*
*preprint arXiv:2201.09802*, 2022.

Felix Berkenkamp, Matteo Turchetta, Angela P Schoellig, and Andreas Krause. Safe model-based reinforcement learning with stability guarantees. *arXiv preprint arXiv:1705.08551*, 2017.

Christopher M Bishop and Nasser M Nasrabadi. *Pattern recognition and*
*machine learning*, volume 4. Springer, 2006.

Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan
Wierstra. Weight uncertainty in neural network. In *International*
*conference on machine learning*, pages 1613–1622. PMLR, 2015.

Léon Bottou. Online learning and stochastic approximations. *On-line*
*learning in neural networks*, 17(9):142, 1998.

Nawaf Bou-Rabee and Martin Hairer. Nonasymptotic mixing of the
mala algorithm. *IMA Journal of Numerical Analysis*, 33(1):80–110,

2013.

Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial time algorithm for near-optimal reinforcement learning. *Journal*
*of Machine Learning Research*, 3(Oct):213–231, 2002.

Sébastien Bubeck, Nicolo Cesa-Bianchi, et al. Regret analysis of
stochastic and nonstochastic multi-armed bandit problems. *Foun-*
*dations and Trends® in Machine Learning*, 5(1):1–122, 2012.

Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. *Statistical Science*, pages 273–304, 1995.

312 probabilistic artificial intelligence

Tianqi Chen, Emily Fox, and Carlos Guestrin. Stochastic gradient
hamiltonian monte carlo. In *International conference on machine learn-*
*ing*, pages 1683–1691. PMLR, 2014.

Sayak Ray Chowdhury and Aditya Gopalan. On kernelized multiarmed bandits. In *International Conference on Machine Learning*, pages
844–853. PMLR, 2017.

Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey
Levine. Deep reinforcement learning in a handful of trials using
probabilistic dynamics models. *Advances in neural information pro-*
*cessing systems*, 31, 2018.

Thomas Cover and Joy Thomas. *Elements of information theory* . Wiley
Interscience, 2006.

Sebastian Curi, Felix Berkenkamp, and Andreas Krause. Efficient model-based reinforcement learning through optimistic policy
search and planning. *arXiv preprint arXiv:2006.08684*, 2020.

Sebastian Curi, Armin Lederer, Sandra Hirche, and Andreas Krause.

Safe reinforcement learning via confidence-based filters. *arXiv*
*preprint arXiv:2207.01337*, 2022.

Marc Deisenroth and Carl E Rasmussen. Pilco: A model-based and

data-efficient approach to policy search. In *Proceedings of the 28th*
*International Conference on machine learning (ICML-11)*, pages 465–472.
Citeseer, 2011.

Marc Peter Deisenroth, A Aldo Faisal, and Cheng Soon Ong. *Mathe-*
*matics for machine learning* . Cambridge University Press, 2020.

David Duvenaud. *Automatic model construction with Gaussian processes* .
PhD thesis, University of Cambridge, 2014.

David Duvenaud and Ryan P Adams. Black-box stochastic variational
inference in five lines of python. In *NIPS Workshop on Black-box Learn-*
*ing and Inference*, 2015.

Eyal Even-Dar and Yishay Mansour. Convergence of optimistic and
incremental q-learning. *Advances in neural information processing sys-*

*tems*, 14, 2001.

Eyal Even-Dar, Yishay Mansour, and Peter Bartlett. Learning rates for
q-learning. *Journal of machine learning Research*, 5(1), 2003.

Karl Friston. The free-energy principle: a unified brain theory? *Nature*
*reviews neuroscience*, 11(2):127–138, 2010.

bibliography 313


Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep bayesian active learning with image data. In *International Conference on Machine*
*Learning*, pages 1183–1192. PMLR, 2017.

Daniel Golovin, Benjamin Solnik, Subhodeep Moitra, Greg Kochanski, John Karro, and David Sculley. Google vizier: A service for
black-box optimization. In *Proceedings of the 23rd ACM SIGKDD in-*
*ternational conference on knowledge discovery and data mining*, pages
1487–1495, 2017.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In *International conference on*
*machine learning*, pages 1321–1330. PMLR, 2017.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.
Soft actor-critic: Off-policy maximum entropy deep reinforcement
learning with a stochastic actor. In *International conference on machine*
*learning*, pages 1861–1870. PMLR, 2018.

Jouni Hartikainen and Simo Särkkä. Kalman filtering and smoothing
solutions to temporal gaussian process regression models. In *2010*
*IEEE international workshop on machine learning for signal processing*,
pages 379–384. IEEE, 2010.

Nicolas Heess, Greg Wayne, David Silver, Timothy Lillicrap, Yuval
Tassa, and Tom Erez. Learning continuous control policies by
stochastic value gradients. *arXiv preprint arXiv:1510.09142*, 2015.

James Hensman, Alexander Matthews, and Zoubin Ghahramani. Scalable variational gaussian process classification. In *Artificial Intelli-*
*gence and Statistics*, pages 351–360. PMLR, 2015.

Thomas Hofmann. *Computational Intelligence Lab* . ETH Zurich, 2022.

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov,
and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. *arXiv preprint arXiv:1803.05407*, 2018.

Tommi Jaakkola, Michael Jordan, and Satinder Singh. Convergence of
stochastic iterative dynamic programming algorithms. *Advances in*
*neural information processing systems*, 6, 1993.

Alex Kendall and Yarin Gal. What uncertainties do we need in

bayesian deep learning for computer vision? *Advances in neural*
*information processing systems*, 30, 2017.

Torsten Koller, Felix Berkenkamp, Matteo Turchetta, and Andreas
Krause. Learning-based model predictive control for safe exploration. In *2018 IEEE conference on decision and control (CDC)*, pages
6059–6066. IEEE, 2018.

314 probabilistic artificial intelligence

Andreas Krause and Daniel Golovin. Submodular function maximiza
tion. *Tractability*, 3:71–104, 2014.

Andreas Krause and Fanny Yang. *Introduction to Machine Learning* .
ETH Zurich, 2022.

Tor Lattimore and Csaba Szepesvári. *Bandit algorithms* . Cambridge
University Press, 2020.

David A Levin and Yuval Peres. *Markov chains and mixing times*, volume
107. American Mathematical Soc., 2017.

Sergey Levine. Reinforcement learning and control as probabilistic
inference: Tutorial and review. *arXiv preprint arXiv:1805.00909*, 2018.

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas
Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra.

Continuous control with deep reinforcement learning. *arXiv preprint*
*arXiv:1509.02971*, 2015.

Yi-An Ma, Yuansi Chen, Chi Jin, Nicolas Flammarion, and Michael I
Jordan. Sampling can be faster than optimization. *Proceedings of the*
*National Academy of Sciences*, 116(42):20881–20885, 2019.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu,
Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller,
Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. *nature*, 518(7540):529–533,

2015.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex
Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray
Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In *International conference on machine learning*, pages 1928–1937.

PMLR, 2016.

Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih.
Monte carlo gradient estimation in machine learning. *J. Mach. Learn.*
*Res.*, 21(132):1–62, 2020.

Kevin P Murphy. Conjugate bayesian analysis of the gaussian distribution. *def*, 1(2 *σ* 2):16, 2007.

Jayakrishnan Nair, Adam Wierman, and Bert Zwart. *The Fundamentals*
*of Heavy Tails*, volume 53. Cambridge University Press, 2022.

George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. An
analysis of approximations for maximizing submodular set functions. *Mathematical programming*, 14:265–294, 1978.

bibliography 315


Vu Nguyen, Sunil Gupta, Santu Rana, Cheng Li, and Svetha Venkatesh.
Regret for expected improvement over the best-observed value and
stopping condition. In *Asian Conference on Machine Learning*, pages
279–294. PMLR, 2017.

Kaare Brandt Petersen, Michael Syskind Pedersen, et al. The matrix
cookbook. *Technical University of Denmark*, 7(15):510, 2008.

Joaquin Quinonero-Candela and Carl Edward Rasmussen. A unifying
view of sparse approximate gaussian process regression. *The Journal*
*of Machine Learning Research*, 6, 2005.

Maxim Raginsky, Alexander Rakhlin, and Matus Telgarsky. Nonconvex learning via stochastic gradient langevin dynamics: a
nonasymptotic analysis. In *Conference on Learning Theory*, pages
1674–1703. PMLR, 2017.

Ali Rahimi, Benjamin Recht, et al. Random features for large-scale
kernel machines. In *NIPS*, 2007.

Rajesh Ranganath, Sean Gerrish, and David Blei. Black box variational
inference. In *Artificial intelligence and statistics*, pages 814–822. PMLR,

2014.

Philip A Romero, Andreas Krause, and Frances H Arnold. Navigating
the protein fitness landscape with gaussian processes. *Proceedings of*
*the National Academy of Sciences*, 110(3):E193–E201, 2013.

Sebastian Ruder. An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*, 2016.

Stuart Russell and Peter Norvig. *Artificial intelligence: a modern ap-*
*proach* . Prentice Hall, 2002.

Grant Sanderson. But what is the fourier transform? a visual introduc
tion, 2018. URL `https://www.youtube.com/watch?v=spUNpyF58BY` .

Bernhard Schölkopf, Ralf Herbrich, and Alex J Smola. A generalized
representer theorem. In *International conference on computational learn-*
*ing theory*, pages 416–426. Springer, 2001.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and
Philipp Moritz. Trust region policy optimization. In *International*
*conference on machine learning*, pages 1889–1897. PMLR, 2015a.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and
Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*, 2015b.

316 probabilistic artificial intelligence

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and
Oleg Klimov. Proximal policy optimization algorithms. *arXiv*
*preprint arXiv:1707.06347*, 2017.

Guy Shani, Joelle Pineau, and Robert Kaplow. A survey of point-based
pomdp solvers. *Autonomous Agents and Multi-Agent Systems*, 27(1):

1–51, 2013.

Satinder Singh, Tommi Jaakkola, Michael L Littman, and Csaba
Szepesvári. Convergence results for single-step on-policy
reinforcement-learning algorithms. *Machine learning*, 38(3):287–308,

2000.

Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias
Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. In *International Conference on Machine*
*Learning*, pages 1015—1022. ICML, 2010.

Bhavya Sukhija, Matteo Turchetta, David Lindner, Andreas Krause,
Sebastian Trimpe, and Dominik Baumann. Scalable safe exploration for global optimization of dynamical systems. *arXiv preprint*
*arXiv:2201.09562*, 2022.

Richard S Sutton and Andrew G Barto. *Reinforcement learning: An*
*introduction* . MIT press, 2018.

Yee Whye Teh, Alexandre H Thiery, and Sebastian J Vollmer. Consistency and fluctuations for stochastic gradient langevin dynamics.
*Journal of Machine Learning Research*, 17, 2016.

Michalis Titsias and Miguel Lázaro-Gredilla. Doubly stochastic variational bayes for non-conjugate inference. In *International conference*
*on machine learning*, pages 1971–1979. PMLR, 2014.

Matteo Turchetta, Felix Berkenkamp, and Andreas Krause. Safe exploration for interactive machine learning. *Advances in Neural Informa-*
*tion Processing Systems*, 32, 2019.

Sattar Vakili, Kia Khezeli, and Victor Picheny. On information gain and
regret bounds in gaussian process bandits. In *International Conference*
*on Artificial Intelligence and Statistics*, pages 82–90. PMLR, 2021.

Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement
learning with double q-learning. In *Proceedings of the AAAI conference*
*on artificial intelligence*, volume 30, 2016.

Max Welling and Yee W Teh. Bayesian learning via stochastic gradient
langevin dynamics. In *Proceedings of the 28th international conference*
*on machine learning (ICML-11)*, pages 681–688. Citeseer, 2011.

bibliography 317


Christopher K Williams and Carl Edward Rasmussen. *Gaussian pro-*
*cesses for machine learning*, volume 2. MIT press, 2006.

Ronald J Williams. Simple statistical gradient-following algorithms for
connectionist reinforcement learning. *Machine learning*, 8(3):229–256,

1992.

Stephen Wright, Jorge Nocedal, et al. Numerical optimization. *Springer*
*Science*, 35(67-68):7, 1999.

Pan Xu, Jinghui Chen, Difan Zou, and Quanquan Gu. Global convergence of langevin dynamics based algorithms for nonconvex optimization. *Advances in Neural Information Processing Systems*, 31, 2018.

Yinyu Ye. The simplex and policy-iteration methods are strongly polynomial for the markov decision problem with a fixed discount rate.
*Mathematics of Operations Research*, 36(4):593–603, 2011.

### *Acronyms*

*a.s.* almost surely, with high probability, with probability

1

*A2C* advantage actor-critic

*BALD* Bayesian active learning by disagreement

*BLR* Bayesian linear regression

*CDF* cumulative distribution function

*DDPG* deep deterministic policy gradients

*DDQN* double deep Q-networks

*DQN* deep Q-networks

*ECE* expected calibration error

*EI* expected improvement

*ELBO* evidence lower bound

*FITC* fully independent training conditional

*GAE* generalized advantage estimation

*GD* gradient descent

*GLIE* greedy in the limit with infinite exploration

*GP* Gaussian process

*GPC* Gaussian process classification

*GRV* Gaussian random vector

*H-UCRL* hallucinated upper confidence reinforcement
learning

*HMC* Hamiltonian Monte Carlo

*HMM* hidden Markov model

*i.i.d.* independent and identically distributed

*iff* if and only if

*KL* Kullback-Leibler

*LAMBDA* Lagrangian model-based agent

*LASSO* least absolute shrinkage and selection operator

*LD* Langevin dynamics

*LMC* Langevin Monte Carlo

*LOTE* law of total expectation

*LOTP* law of total probability

*LOTUS* law of the unconscious statistician


*LOTV* law of total variance

*MAB* multi-armed bandits

*MALA* Metropolis adjusted Langevin algorithm

*MAP* maximum a posteriori

*MC* Monte Carlo

*MCE* maximum calibration error

*MCMC* Markov chain Monte Carlo

*MDP* Markov decision process

*MI* mutual information

*MLE* maximum likelihood estimate

*MLL* marginal log likelihood

*MPC* model predictive control

*MSE* mean squared error

*NLL* negative log likelihood

*OAC* online actor-critic

*PBPI* point-based policy iteration

*PBVI* point-based value iteration

*PDF* probability density function

*PETS* probabilistic ensembles with trajectory sampling

*PI* probability of improvement

*PI* policy iteration

*PILCO* probabilistic inference for learning control

*PMF* probability mass function

*POMDP* partially observable Markov decision process

*PPO* proximal policy optimization

*RBF* radial basis function

*ReLU* rectified linear unit

*RHC* receding horizon control

*RKHS* reproducing kernel Hilbert space

*RL* reinforcement learning

*RM* Robbins-Monro

*SAA* stochastic average approximation

*SAC* soft actor-critic

320 probabilistic artificial intelligence

*SARSA* state-action-reward-state-action

*SG-HMC* stochastic gradient Hamiltonian Monte Carlo

*SGD* stochastic gradient descent

*SGLD* stochastic gradient Langevin dynamics

*SLLN* strong law of large numbers

*SoR* subset of regressors

*SVG* stochastic value gradients

*SWA* stochastic weight averaging

*SWAG* stochastic weight averaging-Gaussian

*Tanh* hyperbolic tangent


*TD* temporal difference

*TD3* twin delayed DDPG

*TRPO* trust-region policy optimization

*UCB* upper confidence bound

*VI* variational inference

*VI* value iteration

*w.l.o.g.* without loss of generality

*w.r.t.* with respect to

*WLLN* weak law of large numbers

### *Index*

*R* max algorithm, 205
*ℓ* 1 -regularization, 36
*ℓ* 2 -regularization, 36
*ℓ* ∞ norm, 185
*ϵ* -greedy, **203**, 259
*σ* -algebra, 2

acceptance distribution, 128
acquisition function, **168**, 258
activation, 140
activation function, 140
active inference, 242

actor, 232
actor-critic method, 232
Adagrad, 36
Adam, 36
adaptive learning rate, 36
additive white Gaussian noise, 50
advantage actor-critic, 233
advantage function, 229
affine transformation, 37
aleatoric uncertainty, 49
almost sure convergence, 26
aperiodic Markov chain, 123
augmented Lagrangian method, 262
automatic differentiation (auto-diff), 142

backpropagation, 142
Banach fixed-point theorem, 185
Barrier function, 263
baseline, 225
batch, 35
batch size, 35


Bayes by Backprop, 145
Bayes’ rule, 20
Bayes’ rule for entropy, 156
Bayesian active learning by disagreement, 165
Bayesian filtering, 61, **63**, 194
Bayesian linear regression, 53
Bayesian logistic regression, 93
Bayesian networks, 11
Bayesian neural network, 143
Bayesian optimization, 170
Bayesian reasoning, 1
Bayesian smoothing, 64
belief, 62, **194**
belief-state Markov decision process, 195
Bellman error, **218**, 238
Bellman expectation equation, 182
Bellman optimality equation, 188
Bellman update, 187
Bellman’s optimality principle, 188
Bellman’s theorem, 187

Bernoulli distribution, 5

bias, 23

bias-variance tradeoff, **23**, 234
binary cross-entropy loss, 103
binomial distribution, 5

black box stochastic variational inference, 117

Bochner’s theorem, 84

Boltzmann distribution, 131
bootstrapping, 149, **207**
Borel *σ* -algebra, 3
bounded discounted payoff, 222
burn-in time, 127

322 probabilistic artificial intelligence

calibration, 149
categorical distribution, 5
central limit theorem, 7

Cesàro mean, 170
chain rule for entropy, 156
chain rule of probability, 9
change of variables formula, 18
Chebyshev’s inequality, 26
Cholesky decomposition, 42
classification, 18
competitive ratio, 170
completing the square, 278
computation graph, 139
concave function, 32
conditional distribution, 9
conditional entropy, 156
conditional expectation, 14
conditional independence, 11
conditional likelihood, 20

conditional linear Gaussian, 41
conditional mutual information, 158
conditional probability, 8
conditional variance, 17
conditioning, 9
conjugate prior, 21
consistent estimator, 23
constrained Markov decision processes, 262
continuous setting, 200
contraction, 185
control, 170
convergence in distribution, 26
convergence in probability, 26
convergence with probability 1, 26
convex body chasing, 170
convex function, 32
convex function chasing, 170
correlation, 15

covariance, 15
covariance function, 69

covariance matrix, 16

critic, 232

cross-entropy, 102
cross-entropy loss, 142
cross-entropy method, 248


d-separation, 12
data processing inequality, 159
deep deterministic policy gradients, 237
deep Q-networks, 220
deep reinforcement learning, 218
density, 6
design matrix, 47
detailed balance equation, 126
deterministic policy gradients (DDPG), 251
differentiation under the integral sign, 13
Dirac delta function, 9
directed graphical model, 11
directional derivative, 32
discounted payoff, **180**, 223, 247
discounted state occupancy measure, 231
double DQN, 220

downstream return, 226

dropout regularization, 147
dynamics model, 179

eligibility vector, 228
empirical Bayes, 80
empirical risk, 19
energy function, 131

entropy, 51, **99**
entropy regularization, 240
entropy-regularized Markov decision process, 240
episode, 200
episodic setting, 200
epistemic uncertainty, 49
epoch, 35
equality with high probability, 26
ergodic theorem, 127
ergodicity, 123

estimator, 22

Euler’s formula, 84

event, 2

event space, 2
evidence, 20

evidence lower bound, 111

exclusive KL-divergence, 107
expectation, 12
expected calibration error, 150
expected improvement, 174

experience replay, 220
experimental design, 155
exploration distribution, 236
exploration noise, 258
exploration-exploitation dilemma, 114, **167**
exponential family, 232
exponential kernel, 73

feature space, 56
feed-forward neural network, 142
finite measure, **120**, 231
finite-horizon, 181
first-order characterization of convexity, 32
first-order expansion, 31
first-order optimality condition, 31
fixed-point iteration, 184
forward KL-divergence, 107
forward sampling, 71
forward-backward algorithm, 65
Fourier transform, 84
free energy, 113
free energy principle, **113**, 242
fully independent training conditional, 90
fully observable environment, 180
function class, 18

function-space view, 57
fundamental theorem of ergodic Markov chains,

124

Gaussian, 6, **37**
Gaussian kernel, 73

Gaussian Markov Process, 88

Gaussian noise “dithering”, 236, **259**
Gaussian process, 69
Gaussian process classification, 93, **97**
Gaussian random vector, 38
generalized advantage estimation, 234
generative model, **20**, 111
Gibbs distribution, 131
Gibbs sampling, 130
Gibbs’ inequality, 105
global optimum, 31
gradient, 13
greedy in the limit with infinite exploration, 203


index 323

greedy policy, 186
Gumbel-max trick, 239

Hadamard’s inequality, 173
hallucinated upper confidence reinforcement
learning, 261
Hamiltonian, 137
Hamiltonian Monte Carlo, 136
heavy-tailed distribution, 24
Hessian, 34

heteroscedastic, **18**

heteroscedastic noise, 143
hidden layer, 139
hidden Markov model, 193
histogram binning, 151
Hoeffding’s inequality, 28
Hoeffding’s lemma, 27
homoscedastic, **18**

homoscedastic noise, 143
hyperbolic tangent, 140
hyperprior, **82**, 177

importance weighting, 234
inclusive KL-divergence, 107
independence, 10
inducing points method, 88
inference, 19
information gain, 157
information never hurts principle, **157**, 158
input layer, 139
inputs, 18

instantaneous, 181

instantaneous regret, 169
interaction information, 159
inverse Fourier transform, 84
inverse transform sampling, 7
irreducible Markov chain, 122

isotonic regression, 151
isotropic Gaussian, 37
isotropic kernel, 75

Jacobian, 18
Jensen’s inequality, 101
joint distribution, 8

324 probabilistic artificial intelligence

joint entropy, 156
joint likelihood, 20

Kalman filter, 61
Kalman gain, 65, **66**
Kalman smoothing, 64
Kalman update, 66
kernel function, 58, 69, **72**
kernel matrix, 58
kernel ridge regression, 78
kernel trick, 58
kernelization, 58
Kolmogorov axioms, 2
Kullback-Leibler divergence, 104

labels, 18

Lagrangian model-based agent, 264
Langevin dynamics, 134
Langevin Monte Carlo, 134
Laplace approximation, 92
Laplace distribution, 25
Laplace kernel, 73
lasso, 53
law of large numbers, 27
law of the unconscious statistician, 13
law of total expectation, 14
law of total probability, 9
law of total variance, 17
layer, 139
Leapfrog method, 137
learning, 19
least squares estimator, 48
length scale, 73
light-tailed distribution, 24
likelihood, 20

linear kernel, 72
linear regression, 47
linearity of expectation, 12
local optimum, 31
log-concave distribution, 131
log-likelihood, 29
log-normal distribution, 25
log-prior, 30
logistic function, 93


logistic loss, 94
logits, 139
loss function, 30

Mahalanobis norm, 41
marginal gain, 160
marginal likelihood, 20
marginalization, 8
Markov chain, 120

Markov decision process, 179
Markov property, 62, **120**
Markov’s inequality, 26
matrix inversion lemma, 98
Matérn kernel, 74

maximization bias, 220

maximizing the marginal likelihood, 79
maximum a posteriori estimate, 30
maximum calibration error, 151
maximum entropy principle, 51
maximum likelihood estimate, 29
mean function, 69
mean payoff, 181
mean square continuity, 74

mean square convergence, 74
mean square differentiability, 74
mean squared error, **23**, 141
Mercer’s theorem, 72
method of moments, 109
metrical task systems, 170
Metropolis adjusted Langevin algorithm, 134
Metropolis-Hastings algorithm, 128
Metropolis-Hastings theorem, 128
mixing time, 124
mode, 6

model predictive control, 247
model-based reinforcement learning, **201**, 246
model-free reinforcement learning, 201
moment matching, **109**, 252
momentum, 36
monotone submodularity, 161
Monte Carlo approximation, **25**, 252
Monte Carlo control, 203
Monte Carlo dropout, 147
Monte Carlo rollouts, 252

Monte Carlo trajectory sampling, 249
Monte Carlo tree search, 248
most likely explanation, 194
multi-armed bandits, 168

multicollinearity, 48
multinomial distribution, 5
mutual information, 157

negative log-likelihood, 29, **142**
neural fitted Q-iteration, 220
neural network, 139
normal distribution, 6, **37**
normalizing constant, 21

objective function, 30
off-policy method, 201
on-policy method, 201
online, 62

online actor-critic, 233
online algorithms, 170
online Bayesian linear regression, 56
online learning, 168
optimal control, 247
optimism in the face of uncertainty principle, **169**,

204, 259
optimistic exploration, 260
optimistic Q-learning, 212
output layer, 139
output scale, 72, **75**
overfitting, 19

partially observable Markov decision process, 193
performance plan, 265
planning, 114, **179**, 199
plate notation, 12
Platt scaling, 151
plausible model, 259
point density, 9
point-based policy iteration, 196
point-based value iteration, 196
pointwise convergence, 87
policy, 180
policy gradient method, 222
policy gradient theorem, 230


index 325

policy iteration, **189**, 235
policy search method, 222
policy value function, 222
Polyak averaging, 220
population risk, 19
positive definite, 41
positive definite kernel, 72
positive semi-definite, 41
posterior, 20, **21**
precision matrix, 37
predictive posterior, 21
prior, 20
probabilistic ensembles with trajectory sampling,
256
probabilistic inference for learning control, 252,

**256**
probability, 3
probability density function, **6**
probability matching, 175
probability measure, **2**
probability of improvement, 174
probability simplex, **5**, 196
probability space, 3
probit likelihood, 97
product rule, 9
proposal distribution, 128
proximal policy optimization, 235
pseudo-random number generators, 7

Q-function, 181

Q-learning, 210
quadratic form, 41
quantile function, 7

radial basis function kernel, 73
random Fourier features, 84
random shooting methods, 247
random variable, 4

random vector, 8

rapidly mixing Markov chain, 125
realizations of a random variable, 4
receding horizon control, 247
rectified linear unit, 140
recursive Bayesian estimation, 61

326 probabilistic artificial intelligence

redundancy, 159
regression, 18
regret, 169
Reichenbach’s common cause principle, 11
REINFORCE algorithm, 227
reinforcement learning, 199
relative entropy, 104
reliability diagram, 150
reparameterizable distribution, 116
reparameterization trick, **115**, 239, 250
replay buffer, 220
representer theorem, 77
reproducing kernel Hilbert space, 76
reproducing property, 77
reverse KL-divergence, 107
reversible Markov chain, 126

reward shaping, 192
reward to go, 226
ridge regression, 48
Robbins-Monro algorithm, 34
Robbins-Monro conditions, 34

robust control, 262

rollout, 223

saddle point, 31
safety filter, 267
safety plan, 265
sample covariance matrix, 24
sample mean, 23
sample space, 2
sample variance, 24
SARSA, 208

score function, 224
score function trick, 224
score gradient estimator, 115, **224**
second-order characterization of convexity, 34
second-order expansion, 33
sharply concentrated, 24
shift-invariant kernel, 75
small tails, 24
snapshot, 146
soft actor critic, **240**, 252
soft value function, 240
softmax exploration, 211


softmax function, 140
spectral density, 85

square root, 42
squared exponential kernel, 73
standard deviation, 15
standard normal distribution, 6, **37**
state of a random variable, 4
state space model, 61
state value function, 181

state-action value function, 181

state-dependent baseline, 226
stationary distribution, 122
stationary kernel, 75
stationary point, 31
stochastic average approximation, 249
stochastic environment, 179
stochastic gradient descent, 35
stochastic gradient Langevin dynamics, 135
stochastic matrix, 121

stochastic process, 120
stochastic semi-gradient descent, 217
stochastic value gradients, 239
stochastic weight averaging, 147
stochastic weight averaging-Gaussian, 147
strict convexity, 32
sub-Gaussian random variable, 27
sublinear regret, 169
submodularity, 161
subsampling, 146
subset of regressors, 89
sufficient statistic, 146
sum rule, 8

supervised learning, **18**, 155

support, 5
supremum norm, 185
surprise, 98

synergy, 159

tabular setting, 199
tail distribution function, 24

target space, 4
temperature scaling, 151
temporal models, 67
temporal-difference error, 217

temporal-difference learning, 207
testing conditional, 89
Thompson sampling, **176**, 259
time-homogeneous process, 120
total variation distance, 124
tower rule, 14
training conditional, 89
trajectory, 200
trajectory sampling, 249

transition, 200

transition function, 120

transition graph, 121

transition matrix, 121

trust-region policy optimization, 234
twin delayed DDPG, 236
two-filter smoothing, 65

unbiased estimator, 23
uncertainty sampling, 162
uncorrelated, 15
uniform convergence, 87


index 327

uniform distribution, 5, **7**

union bound, 3
universal approximation theorem, 140
universality of the uniform, 7
upper confidence bound, 171

value iteration, 191

variance, 16

variational family, 103
variational inference, **103**, 252
variational parameters, 91
variational posterior, 91
Viterbi algorithm, 194

weak law of large numbers, 27
weight decay, 36
weight-space view, 50
Weinstein-Aronszajn identity, 173
Wiener process, **74**, 120
Woodbury matrix identity, 56

