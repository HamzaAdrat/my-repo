---
title: "Point process discrimination according to repulsion"
# subtitle: "Example based on the myst system"
author: "Hamza ADRAT and Laurent DECREUSEFOND"
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Point process discrimination according to repulsion

## Abstract
In numerous applications, cloud of points do seem to exhibit *repulsion* in the intuitive sense that there is no local cluster as in a Poisson process. Motivated by data coming from cellular networks, we devise a classification algorithm based on the form of the Voronoi cells. We show that, in the particular set of data we are given, we can retrieve some repulsiveness between antennas, which was expected for engineering reasons.

## Introduction
In the performance analysis of cellular systems, the locations of antennas (or base stations) play a major role (see {cite}`BaccelliStochasticGeometryWireless2008`). It is usually admitted that they can be modeled by a Poisson process. But the data which can be gathered from the Web site of the French National Agency of Radio Frequencies, Cartoradio, see {cite}`ANFR`, tend to prove that this may not be the case. More precisely, if we look at the global picture of all antennas in Paris, we see features reminiscent of a Poisson process (local clusters for instance), see {numref}`paris-orange-fig`(left). However, if we look closer and finer, by specifying a region and a frequency band, we see that the antennas locations do seem to exhibit some repulsion (see {numref}`paris-orange-fig`, right picture).

```{figure} /paris-orange.png
---
name: paris-orange-fig
---
Left: Antennas in Paris. Right: Antennas in one frequency  band only.
:label: paris
```

In previous papers, point processes with repulsion have been used to model such systems {cite}`Deng2014`, {cite}`Miyoshi2016`, {cite}`Gomez2015` for no reason but a mere resemblance between the pictures like the right picture in {numref}`paris-orange-fig` and those obtained by simulating a point process with repulsion. The question is then to decide, given one sample of positions of base stations in a bounded domain, whether it is more likely to be modeled by a point process with repulsion or by a *neutral* point process, i.e. where the locations could be considered as coming from independent drawings of some identically distributed random variables. As we only have a single realization,  we cannot use frequency methods. Since the observation window is finite, we cannot either resort to estimates based on stationarity or ergodicity and  we must take care from the side effects.

The rationale behind our work comes from {cite}`goldman_palm_2010`. It is shown there  that the Voronoi cells of the Ginibre point process (a particular point process with repulsion, see below for the exact definition) are in some sense more regular (closer to a circle) than those of a Poisson process (see {eq}`theorem_goldman` in Theorem 1.). By simulation, this feature seem to persist for other point processes with repulsion, like Gibbs processes. It is this aspect that we use to construct our classification algorithm.
We will simulate several configurations (repulsive and non-repulsive) with the same given  number of points $N$. For each configuration, we will compute the Voronoi diagrams and construct two vectors which will represent the input of our algorithm; an area vector containing the areas of the $10$ innermost Voronoi cells in order to avoid edge effects, plus $4$ other average areas from $20$ cells to have more information on the configuration. And a second perimeter vector which is constructed in the same way, containing the squared perimeters of the corresponding Voronoi cells.
The choice of areas and square perimeters as aspects to our classification task is based on the *isoperimetric inequality in $\mathbf{R}^2$ that states, for the length $P$ of a closed curve and the area $A$ of the planar region that it encloses, that
```{math}
:label: isoperimetric_inequality
P^2 \ge 4 \pi A
```
and that equality holds if and only if the curve is a circle. After normalization, we test some classical ML models (logistic regression, random forest, support vector machine, XGBoost) to classify between repulsive and neutral point processes. The results are surprisingly good even though we trained our models only on Ginibre point processes to represent the whole family of point processes with repulsion.

This paper is organized as follows. We first recall the theoretical notions that we will need in the rest of this paper. We will also briefly define the Papangelou intensity which is at the core of the definition of repulsion. In section 3 we show numerically, and based on two Machine Learning classification models, how the locations of antennas in Paris can be considered as repulsive configurations.

This document provides a Myst template for contributions to the **Computo**
Journal {cite}`computo`. We show how `Python` {cite}`perez2011python`, `R`, or `Julia` code can be included.
Note that you can also add equations easily:

$$
\sum x + y = Z
$$

You can also add equations in such a way to be able to reference them later:

```{math}
:label: math
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

See equation {eq}`math` for details.

## Preliminaries

 A configuration on $E=\mathbf R^2$ is a locally finite (respectively finite) subset of $E$. The space of configurations (respectively finite configurations) is denoted $\mathfrak N$ (respectively $\mathfrak N_{f}$). We equip $\mathfrak N$ with the topology of vague convergence, under which it is a complete, separable, metric space. We denote by $\mathcal B(\mathfrak N)$ the Borelean $\sigma$-field on $\mathfrak N$. A locally finite (respectively finite) point process is a random variable with values in $\mathfrak N$ (respectively $\mathfrak N_{f}$).

 **Definition 1.**
 Let $\Phi$ be a locally finite point process on $E$. Its *correlation functions* $\rho^{(k)} \colon \mathfrak N_{f} \to \mathbb R_+$ are given for any measurable function $f \colon \mathfrak N_{f} \to \mathbb R_+ $ by:

$$
\mathbb{E}\left[ \sum_{\substack{\alpha \in \mathfrak N_{f} \\ \alpha \subset \Phi}} f(\alpha) \right] = \sum_{k=1}^{+ \infty} \frac{1}{k!} \int_{(E)^k} f(\{x_1, \dots, x_k\}) \rho^{(k)}(\{x_1, \dots, x_k\}) \, d x_1 \ldots d x_k .
$$

It is however easier to work with the so-called Janossy measures, whose links with correlation functions are given in {cite}`Daley2003`.

**Definition 2.**
Let $\Phi$ be a finite point process on $E$. Its *Janossy measure* $J$ is given for any $A \in \mathcal B(\mathfrak N_f)$ by:

$$
\mathbb{P}(\Phi \in A) = \sum_{k=1}^{+ \infty} \frac{1}{k!} J(A^{(k)}),
$$

where, for any $k \in \mathbb N^*, \; A^{(k)} = \{ \phi \in A,\; \phi(E) = k \}$.

**Definition 3.**
Let $\Phi$ be a finite point process on $E$. It is said to be regular if there exist *Janossy functions* $(j^{(k)}, k\ge 0)$ such that for any measurable $f \, \colon \, \mathfrak N_{f}\to \mathbb R^{+}$, we have

```{math}
:label: janossy_functions
E[ f(\Phi) ] =  \sum_{n\ge 0} \frac{1}{n!} \int_{E^n} f(\{x_1,\dots,x_n\}) \, j^{(n)}(\{x_1,\dots,x_n\}) d x_1 \ldots d x_n.
```

With this definition in hand, we can define the concept of repulsion , following {cite}`Georgii2005`.

**Definition 4.**
For $\Phi$  a finite point regular process on $E$ with Janossy functions $(j^{(k)},\, k \ge 0)$, its Papangelou intensity $c$ is given for any $x \in E$ and $\phi \in \mathfrak N_{f}$ by:

$$
c(x, \phi) = \frac{j^{(k+1)}(\{x\}\cup \phi)}{j^{(k)}(\phi)} \, \mathbf{1}_{\{j^{(k)}(\phi) \ne 0\} } \text{ if } \phi(E)=k.
$$

The quantity $c(x,\phi)$ can be intuitively thought as the probability to have a particle at $x$ given the observation $\phi$. Intuitively a point process shows repulsion when $\phi \mapsto c(x,\phi)$ is, in some sense, decreasing:

**Definition 5.**
A point process $\Phi$ on $E$ with a version $c$ of its Papangelou intensity is said to be *repulsive* if, for any $\omega, \phi \in \mathfrak N_{f}$ such that $\omega \subset \phi$ and any $x \in E$,

$$
c(x,\phi) \le c(x, \omega).
$$

With this definition, it is not hard to see that Gibbs point processes, determinantal point processes are repulsive (see {cite}`Georgii2005`, {cite}`HoughDeterminantalprocessesindependence2006`). We will not dwell into the vast literature about determinantal point processes, we only focus on the so-called Ginibre point process. It has the interesting feature that we know its distribution when restricted to a compact ball in $E$. For reasons which will be self-evident, we identify hereafter $\mathbb R^{2}$ and $\mathbb C$.

**Definition 6.**
The *Ginibre point process* with intensity $\rho = \frac{\lambda}{\pi}$ (with $\lambda > 0$) is a locally finite point process on $\mathbb C$ that can be defined by its correlation functions:

```{math}
:label: correlation_functions_determinantal
\rho^{(k)}(x_1, \dots, x_k) = \det(K(x_i, x_j), \; 1\le i,j \le k)
```
where $K$ is given by:

$$
K(x,y) = \rho \, e^{-\frac{\lambda}{2}(|x|^2 + |y|^2)}e^{\lambda x \bar{y}},\ \forall \, (x,y) \in \mathbb C^2.
$$

When restricted to the $B(0,R)$, the Ginibre point process admits correlation functions of the form {eq}`correlation_functions_determinantal` with $K=K_R$ given by:

```{math}
:label: eq_main:1
K_R(x,y)=\sum_{j=1}^{+\infty} \frac{\gamma(j+1,R^2)}{j!} \phi_j(x)\phi_j(\bar y)
```
with

$$
\phi_j(x)=\sqrt{\frac{\rho}{\gamma(j+1,R^2)}} \left(\sqrt{\lambda}x\right)^j\, e^{-\frac{\lambda}{2} |x|^2},
$$

and $\gamma(n,x)$ is the lower incomplete Gamma function.

The simulation of such a point process is a delicate matter, first solved in {cite}`HoughDeterminantalprocessesindependence2006`. It remains costly because the algorithm contains complex calculations and some rejections. In order to fasten the procedure, an approximate algorithm has been given in {cite}`MR4279876` (see the bibliography therein to get the URL of the Python code).

For an at most denumerable set of points $\{x_{n}, \, n\ge 1\}$, the Voronoi cells are defined as the convex sets

$$
\mathcal{C}(x_{i})=\{z\in \mathbb C,\ |z-x_{i}|\le |z-x_{j}|  \text{ for all }j\neq i\}.
$$

When the points are drawn from a point process, we thus have a collection of random closed sets. When the process under consideration is stationary with respect to translations, it is customary to define the typical law of a Voronoi cell as the law of the cell containing the origin of $\mathbb R^{2}$ when the point process is taken under its Palm distribution {cite}`goldman_palm_2010`, {cite}`BaccelliStochasticGeometryWireless2009`. It turns out that we know the Palm distribution of the Poisson process (which is itself) and of the Ginibre point process (the correlation functions are of the form {eq}`correlation_functions_determinantal` with $K$ being $K_{R}$ with the first term removed).
We denote by $\mathcal{C}_p$ (respectively $C_{G}$) the typical cell of the Voronoi tessellation associated to a stationary Poisson process in $\mathbb C$ with intensity $\lambda$ (respectively to the Ginibre point process of intensity $\rho$). One of the main theorems of {cite}`goldman_palm_2010` is the following.

**Theorem 1.**
When $r \to 0,$
```{math}
:label: theorem_goldman
\mathbb{E} \left[ V(\mathcal{C}_{G} \cap B(0,r)) \right] = \mathbb{E} \left[ V(\mathcal{C}_p \cap B(0,r)) \right] (1 + r^2 W + \circ(r^2))
```
where $W$ is a positive random variable.

This theorem shows that near the germs of the cells a more important part of the area is captured in the Ginibre–Voronoi tessellation than in the Poisson–Voronoi tessellation. This is an indication that the Voronoi cells of the Ginibre point process are more circular than those given by the Poisson process. This can be corroborated by simulation as shows the {numref}`voronoi-fig`

```{figure} /Voronoi.png
---
name: voronoi-fig
---
On the left, Voronoi associated to a realization of a Poisson process. On the right, Voronoi cells associated to a realization of a Ginibre point process.
```

As we know that circles saturate the isoperimetric inequality, it is sensible to consider classification algorithms based on area and squared perimeter of Voronoi cells. In order to avoid side effects, we concentrate on the innermost cells of the observation window.

## Classification on CARTORADIO data
The Cartoradio web site contains the locations (in GPS coordinates) and other informations about all the antennas (or base stations) in metropolitan France for any operator, any frequency band and all generation of wireless systems (2G to 5G). The capacity of an antenna depends on its power and on the traffic demand it has to serve.  Outside metropolitan areas, the antennas are relatively scarce and located along the main roads to guarantee a large surface coverage (around 30 km$^2). Hence there is no model which can be constructed for these regions.  In big towns, the density of base stations is much higher to handle the traffic demand: An antenna covers around half a squared kilometer. This is thus where the dimensioning problem do appear. One should have a sufficient number of antennas per unit of surface to transport all the traffic, on the other hand, base stations operating in a given frequency band cannot be to close to mitigate interference. This explains the right picture of Figure {fig}`paris`.

When it comes to assess the type of point process we should consider, we cannot consider the city as a whole: the geography (notably the Seine river in Paris, the parks, etc.), the non uniformity of demands (the traffic is heavier aroung railway stations or touristic sites,  for instance) which entails a higher density of antennas,  ruin any kind of invariance a statistician could hope for. That means, we should restrict our expectations to local models of a the size district or a bit more. Since interference, which are the main annoyance to be dealt with, are a local phenomenon, working on a partial part of the whole domain is sufficient to predict and dimension a wireless network. 



Since this type of raw data needed for our task is not already available somewhere, we will need to create the data ourselves, which is some different configurations that are either repulsive or not. Then we make transformations on this raw data in order to extract the information and feed it to the models. The same information will be extracted from the CARTORADIO data so we can test it on our models.

### Data construction

The goal of this section is to create the raw data needed for our classification task, and to which we will apply some transformations in order to extract the final data that will be used to train our models.

The idea is for a given number $N$, we will create many finite configurations of $N$ points each that are either repulsive (Ginibre configuration for example) or not (Poisson configuration). So we will start by calling the ginibre sampling algorithm mentionned earlier in {cite}`MR4279876` in order to generate Ginibre configurations.
For the other type of configurations, i.e. non repulsive ones, we will simulate an homogeneous Poisson point process on the circle $\mathcal{C}(0,r)$, where the radius $r = \sqrt{N}$.

*Remark:* In the Ginibre simulation, we will use a radius $r = E(\sqrt{N})$, where $E(x)$ is the integer of $x$, in order to avoid the problems (and therefore the misleadings) in the edge of the circles.

For each of the previous samples, we compute the Voronoi diagrams and we first extract the areas of these cells. But since we will have different configurations with different cardinals $N$, we will order the cells by proximity of their barycenters to the origin, and then extract the areas of the first ten cells, the idea of having this small number of areas is that we consider that these $10$ cells contain the main information needed for our classification. In addition to that, we will compute the mean of the first $5$, $10$, $15$ and $20$ cells' areas in order to have more information that can be significant for our models.

Then in the same time, we extract the square perimeters of the same cells already used so that, hopefuly, the models could exploit the isoperimetric inequality and separate the two type of configurations.

The final data will be a set of observations where each one contains $29$ columns described as follow:
- For $1 \le i \le 10, \; \mathrm{V}_i$ is the area of the $i^{\mathrm{th}}$ Voronoi cell.
- For $1 \le j \le 10, \; \mathrm{P}_j$ is the square perimeter of the $j^{\mathrm{th}}$ Voronoi cell.
- For $k \in \{5, 10, 15, 20\}, \; \mathrm{MV}_{k}$ (respectively $\mathrm{MP}_{k}$) is the average area (respectively square perimeter) of the first $k$ Voronoi cells.
- A binary column that represent the type of the initial configuration ($1$ for repulsive/Ginibre - $0$ for non-repulsive/Poisson).

The final column will be the target variable for our classification models.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(np.arange(10))
```

```{code-cell} python3
---
tags: [show-output, show-input]
---
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Delaunay, Voronoi, ConvexHull

font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 11,}

# Useful functions for creating the data:

def convert_complex_points(l):
    return l.real, l.imag

def convert_lists_to_points(l1, l2):
    return np.array([l1, l2]).T

def extract_Voronoi_areas(vor):  
    areas, perim = [], []
    for i in range(len(vor.filtered_regions)):
        areas.append(round(ConvexHull(vor.vertices[vor.filtered_regions[i], :]).volume, 3))
        perim.append(ConvexHull(vor.vertices[vor.filtered_regions[i], :]).area)
    return areas, list(np.around((np.array(perim))**2, 3))

def in_box(towers, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= towers[:, 0], towers[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= towers[:, 1], towers[:, 1] <= bounding_box[3]))


def voronoi(towers, bounding_box, N):
    # Select towers inside the bounding box
    i = in_box(towers, bounding_box)
    # Mirror points
    points_center = towers[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left, points_right, axis=0),
                                 np.append(points_down, points_up, axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = Voronoi(points)
    # Filter regions
    regions = []
    [vor.point_region[i] for i in range(N)]

    vor.filtered_points = points_center
    vor.filtered_regions = [vor.regions[vor.point_region[i]] for i in range(len(points_center))]
    return vor

def dpp_Moroz(N):
    radius = int(np.sqrt(N)) ; precision = 2**-53 ; error = False ; quiet=True ; output=None 
    args = [radius, N, kernels['ginibre'], precision, error, quiet, output]
    
    X_dpp_Mz, Y_dpp_Mz = convert_complex_points(sample(*args))
    X_dpp_Mz = X_dpp_Mz*((np.sqrt(N))/radius) ; Y_dpp_Mz = Y_dpp_Mz*((np.sqrt(N))/radius)
    dpp_Mz_points = convert_lists_to_points(X_dpp_Mz, Y_dpp_Mz)
    
    indices = np.argsort((dpp_Mz_points[:,0])**2 + ((dpp_Mz_points[:,1])**2))
    dpp_Mz_points = dpp_Mz_points[indices]
    
    dpp_Mz_vor = voronoi(dpp_Mz_points, (-np.sqrt(N)-.1, np.sqrt(N)+.1, -np.sqrt(N)-.1, np.sqrt(N)+.1), len(dpp_Mz_points))
    Voronoi_areas, Voronoi_perim = extract_Voronoi_areas(dpp_Mz_vor)
    
    return [Voronoi_areas, Voronoi_perim, 1]

def random_process(N):
    radius = np.sqrt(N)
    alpha = 2 * np.pi * scipy.stats.uniform.rvs(0,1,N)
    r = radius * np.sqrt(scipy.stats.uniform.rvs(0,1,N))
    
    X_rand, Y_rand = r*np.cos(alpha), r*np.sin(alpha)
    rand_points = convert_lists_to_points(X_rand, Y_rand)
    
    indices = np.argsort((rand_points[:,0])**2 + ((rand_points[:,1])**2))
    rand_points = rand_points[indices]
    
    rand_vor = voronoi(rand_points, (-radius -.1, radius +.1, -radius -.1, radius +.1), len(rand_points))
    Voronoi_areas, Voronoi_perim = extract_Voronoi_areas(rand_vor)
    
    return [Voronoi_areas, Voronoi_perim, 0]

def create_dataframe(N, observations):
    list_df = []
    for i in range(observations):
        list_df.append(dpp_Moroz(N))
        list_df.append(random_process(N))
    df = pd.DataFrame(list_df, columns =['Voronoi_areas', 'Voronoi_perim', 'Type'])
    return df

def normalize(vec):
    vec = np.array(vec)
    m, e = np.mean(vec), np.sqrt(np.var(vec))
    return (vec - m)/e

def compute_mean(l, n):
    if n <= len(l):
        new_l = np.array(list(list(zip(*l))[:n])).T
    else:
        new_l = np.array(list(list(zip(*l))[:len(l)])).T

    return np.mean(new_l, axis=1)

def single_area(l, k):
    l = np.array(l)
    return l[:,k]

def transform_df(odf):
    
    list_V = odf['Voronoi_areas'].tolist()
    list_P = odf['Voronoi_perim'].tolist()
    
    [MV5, MV10, MV15, MV20] = [compute_mean(list_V, n) for n in [5, 10, 15, 20]]
    [MP5, MP10, MP15, MP20] = [compute_mean(list_P, n) for n in [5, 10, 15, 20]]
    
    normalized_V10 = [normalize(list_V[i][:10]) for i in range(odf.shape[0])]
    normalized_P10 = [normalize(list_P[i][:10]) for i in range(odf.shape[0])]
    
    [V1, V2, V3, V4, V5, V6, V7, V8, V9, V10] = [single_area(normalized_V10, k) for k in range(10)]
    [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10] = [single_area(normalized_P10, k) for k in range(10)]
    
    dict_df = {'V1':V1, 'V2':V2, 'V3':V3, 'V4':V4, 'V5':V5, 'V6':V6, 'V7':V7, 'V8':V8, 'V9':V9, 'V10':V10,
               'MV5':MV5, 'MV10':MV10, 'MV15':MV15, 'MV20':MV20,
               'P1':P1, 'P2':P2, 'P3':P3, 'P4':P4, 'P5':P5, 'P6':P6, 'P7':P7, 'P8':P8, 'P9':P9, 'P10':P10,
               'MP5':MP5, 'MP10':MP10, 'MP15':MP15, 'MP20':MP20,
               'type': odf['Type']}
    
    return pd.DataFrame(dict_df)

%run -i Moroz_dpp.py
```

Here is an example of the data created. We generate a data of $3000$ observations of configurations ($1500$ repulsive and $1500$ non repulsive) of $N = 24$ points.

```{code-cell} python3
---
tags: [show-output, show-input]
---
ddf = create_dataframe(24, 1500)
ddf_transformed = transform_df(ddf)
ddf_transformed.head()
```

## Classifictaion models

In this section, we will train and test some Machine Learning models using the data we've created in the previous section. For a start we will select all the columns as inputs to our models (this can lead to false predictions, especially if some columns share the same information). And also we will only use baseline models, i.e. all the hyperparameters' values are taken as defaults, (a grid search can be used later in order to select the optimal hyperparameters for each model).

The models we use in this paper are:
- Logistic regression, which is a classification algorithm, used when the value of the target variable is categorical in nature. Logistic regression is most commonly used when the data in question has binary output, so when it belongs to one class or another, or is either a $0$ or $1$.
- Random forest, a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the accuracy and control over-fitting.

Two other classification models (Support Vector Machine and XGBoost) have been tested but not introduced in this paper since they gave similar results to the models prensent in this paper.


(subsec:subheading)=
### This is another subheading

As seen in [section](subsec:subheading), lorem ipsum dolor sit amet,
consectetur adipiscing elit. Cras nec ornare urna. Nullam consectetur

```{table} My table title
:name: my-table-ref

| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |
```

Now we can reference the table in the text (See {ref}`my-table-ref`).


## Discussion

- This is a list
- With more elements
- It isn't numbered.

But we can also do a numbered list

1. This is my first item
2. This is my second item
3. This is my third item

## Conclusion 

```{bibliography}
:style: unsrt
```
