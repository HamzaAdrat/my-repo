---
title: "Point process discrimination according to repulsion"
# subtitle: "Example based on the myst system"
author: "Hamza ADRAT and Laurent DECREUSEFOND"
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Abstract
In numerous applications, cloud of points do seem to exhibit *repulsion* in the intuitive sense that there is no local cluster as in a Poisson process. Motivated by data coming from cellular networks, we devise a classification algorithm based on the form of the Voronoi cells. We show that, in the particular set of data we are given, we can retrieve some repulsiveness between antennas, which was expected for engineering reasons.

# Introduction
In the performance analysis of cellular systems, the locations of antennas (or base stations) play a major role (see {cite}`BaccelliStochasticGeometryWireless2008`). It is usually admitted that they can be modeled by a Poisson process. But the data which can be gathered from the Web site of the French National Agency of Radio Frequencies, Cartoradio, see {cite}`ANFR`, tend to prove that this may not be the case. More precisely, if we look at the global picture of all antennas in Paris, we see features reminiscent of a Poisson process (local clusters for instance), see {numref}`paris-orange-fig` (left). However, if we look closer and finer, by specifying a region and a frequency band, we see that the antennas locations do seem to exhibit some repulsion (see {numref}`paris-orange-fig`, right picture).

```{figure} /paris-orange.png
---
name: paris-orange-fig
---
Left: Antennas in Paris. Right: Antennas in one frequency  band only.

```
In previous papers, point processes with repulsion have been used to model such systems {cite}`Deng2014`, {cite}`Miyoshi2016`, {cite}`Gomez2015` for no reason but a mere resemblance between the pictures like the right picture in {numref}`paris-orange-fig` and those obtained by simulating a point process with repulsion. The question is then to decide, given one sample of positions of base stations in a bounded domain, whether it is more likely to be modeled by a point process with repulsion or by a *neutral* point process, i.e. where the locations could be considered as coming from independent drawings of some identically distributed random variables. As we only have a single realization,  we cannot use frequency methods. Since the observation window is finite, we cannot either resort to estimates based on stationarity or ergodicity and  we must take care from the side effects.

The rationale behind our work comes from {cite}`goldman_palm_2010`. It is shown there that the Voronoi cells of the Ginibre point process (a particular point
process with repulsion, see below for the exact definition) are in some sense more regular (closer to a circle) than those of a Poisson process (see {eq}`theorem_goldman` in Theorem 1.). By simulation, this feature seems to persist for other point processes with repulsion, like Gibbs processes. In {cite}`Taylor2012`, the surface of Voronoi cells is claimed to be a good discrepancy indicator between Poisson process and several processes with repulsion (Gibbs processes, Strauss processes with repulsion and the Geyer saturation model). For any of these models, we do not have any closed formula on the surface of the Voronoi cells so the procedure proposed in this paper is to simulate a large number of realizations of each of these processes and compute the empirical mean and variance of the Voronoi cells area. They obtain mixed conclusions as this sole indicator does not enable to rule out the Poisson hypothesis for many situations.

Our contribution is to consider the ratio of the surface by the squared perimeter instead of the surface of the Voronoi cells alone. Actually, we can interpret
the result of {cite}`goldman_palm_2010` by saying that the Voronoi cells of a Ginibre point process are more circular than those of a Poisson point process. The isoperimetric inequality stands for any regular enough domain in the plane, $R = \frac{4 \pi S}{P^2}$ is less than $1$ and the equality is obtained for disks. It is thus sensible to think that the ratio $R$ will be closer to $1$ for repulsive processes than for neutral point processes. Following the procedure of {cite}`Taylor2012`, we show that we get a much better indicator by using $R$ instead $S$ alone to discriminate between repulsive and neutral point processes.

However, for the application we have in mind, which is to decide for one single map  which model is the most pertinent, we cannot use this criterion based on probability. That is why we resort to an ML model. After several tries, we concluded that the most efficient algorithm was to use Logistic Regression. In a first step, we trained it on simulations of Ginibre and Poisson point processes. The advantage of the Ginibre process is that we have efficient algorithm to simulate it {cite}`MR4279876` and it does not seem to alter the accuracy of our algorithm to use one single class of repulsive point process. We remarked that we obtain a much better discrimination by considering the mean value of $R$ for the five most central cells instead of just the most central one. We can even improve our discrimination rate by adding to the input vector the value of each of the five ratios.

Furthermore, the repulsion in the Ginibre class of point processes can be also modulated by making a $\beta$-thinning (to weaken the repulsion) and then a $\sqrt{\beta}$-dilation (to keep the same intensity of points per surface unit) to obtain what is called a $\beta$-Ginibre. For $\beta=1$, we have the original Ginibre process and when $\beta$ goes to $0$, it tends in law to a Poisson process (see {cite}`DecreusefondAsymptoticssuperpositionpoint2015`) so that we have a full scale of point processes with intermediate repulsion between $0$ and $1$. We show that our logistic regression algorithm can still accurately discriminate between Poisson and $\beta$-repulsive point processes for $\beta$ up to $0.7$.

The paper is organized as follows. We first remind what is a Ginibre point process and the property of its Voronoi cells which motivates the sequel.

# Preliminaries
We consider finite point processes on a bounded window $E$. The law of a such a point process $N$ can be  characterized by its correlation functions (for
details we refer to {cite}`Daley2003`[Chapter 5]). These are symmetric functions $(\rho_{k},k\ge 1)$ such that for any bounded function $f$, we can write:

$$
 \mathbb{E}\left[ \sum_{\alpha \subset N} f(\alpha) \right] = \sum_{k=1}^{+ \infty} \frac{1}{k!} \int_{E^k} f(\{x_1, \dots, x_k\}) \rho_{k}(x_1, \dots, x_k) \, d x_1 \dots d x_k .
$$

Intuitively speaking, $\rho_{k}(x_{1}, \dots, x_{k}) \, d x_{1} \dots d x_{k}$ represents the probability to observe in $N$, at least $k$ points located around the 
point $x_{j}$. For a Poisson point process of control measure $m(x) \, dx$, we have

$$
\rho_{k}(x_{1}, \dots, x_{k}) = \prod_{j=1}^{k} m(x_{j}).
$$

The **Ginibre point process**, restricted to $E=B(0,r)$, with intensity $\rho = \frac{\lambda}{\pi}$ (with $\lambda > 0$) has correlation functions (see {cite}`Decreusefond_2015`)

```{math}
:label: correlation_functions_determinantal
\rho_{k}(x_1, \dots, x_k) = \det(K(x_i, x_j), \; 1 \le i,j \le k)
```
where $K$ is given by

```{math}
:label: eq_main
K_r(x,y)=\sum_{j=1}^\infty \frac{\gamma(j+1,r^2)}{j!} \phi_j(x)\phi_j(\bar y)
```
with

$$
\phi_j(x) = \sqrt{\frac{\rho}{\gamma(j+1,r^2)}} \left(\sqrt{\lambda} x \right)^j \, e^{-\frac{\lambda}{2} |x|^2}
$$

and $\gamma(n,x)$ is the lower incomplete Gamma function. The simulation of such a point process is a delicate matter, first solved in {cite}`HoughDeterminantalprocessesindependence2006`. It remains costly because the algorithm contains complex calculations and some rejections. In order to fasten the procedure, an approximate algorithm, with error estimates, has been given in {cite}`MR4279876` (see the bibliography therein to get the URL of the Python code).

For an at most denumerable set of points $\{x_{n}, \, n \ge 1\}$, the Voronoi cells are defined as the convex sets

$$
\mathcal{C}(x_{i})=\{z \in \mathbb{C},\ |z-x_{i}|\le |z-x_{j}|  \text{ for all }j\neq i\}.
$$

When the points are drawn from a point process, we thus have a collection of random closed sets. When the process under consideration is stationary with respect to translations, it is customary to define the typical law of a Voronoi cell as the law of the cell containing the origin of $\mathbb{R}^{2}$ when the point process is taken under its Palm distribution {cite}`goldman_palm_2010`, {cite}`BaccelliStochasticGeometryWireless2009`. It turns out that we know the Palm distribution of the Poisson process (which is itself) and of the Ginibre point process (the correlation functions are of the form {eq}`correlation_functions_determinantal` with $K$ being $K_{R}$ with the first term removed). 
We denote by $\mathcal{C}_p$ (respectively $\mathcal{C}_{G}$) the typical cell of the Voronoi tessellation associated to a stationary Poisson process in $\mathbb{C}$  with
intensity $\lambda$ (respectively to the Ginibre point process of intensity $\rho$). One of the main theorems of {cite}`goldman_palm_2010` is the following.

**Theorem 1.**
When $r \to 0,$
```{math}
:label: theorem_goldman
\mathbb{E} \left[ V(\mathcal{C}_{G} \cap B(0,r)) \right] = \mathbb{E} \left[ V(\mathcal{C}_p \cap B(0,r)) \right] (1 + r^2 W + \circ(r^2))
```
where $W$ is a positive random variable.

This theorem shows that near the germs of the cells a more important part of the area is captured in the Ginibre–Voronoi tessellation than in the Poisson–Voronoi tessellation. This is an indication that the Voronoi cells of the Ginibre point process are more circular than those given by the Poisson process. This can be corroborated by simulation as shows the Figure {numref}`voronoi-fig`

```{figure} /Voronoi.png
---
name: voronoi-fig
---
On the left, Voronoi cells associated to a realization of a Ginibre process. On the right, Voronoi cells associated to a realization of a Poisson process.
```

As we know that circles saturate the isoperimetric inequality, it is sensible to consider classification algorithms based on area and squared perimeter of Voronoi cells. In order to avoid side effects, we concentrate on the innermost cells of the observation window.

# Classification of CARTORADIO data
The Cartoradio web site contains the locations (in GPS coordinates) and other informations about all the antennas (or base stations) in metropolitan France for any operator, any frequency band and all generation of wireless systems (2G to 5G). The capacity of an antenna depends on its power and on the traffic demand it has to serve.  Outside metropolitan areas, the antennas are relatively scarce and located along the main roads to guarantee a large surface coverage (around 30 km$^2$). Hence there is no need to  construct models for these regions.  On the contrary, in big towns, the density of base stations is much higher to handle the traffic demand: An antenna covers around half a squared kilometer. This is  where the dimensioning problem do appear. One should have a sufficient number of antennas per unit of surface to transport all the traffic, on the other hand, base stations operating in a given frequency band cannot be to close to mitigate interference. This explains the right picture of Figure {numref}`paris-orange-fig`.

When it comes to assess the type of point process we should consider in this situation, we cannot consider the city as a whole: the geography (notably the Seine river in Paris, the parks, etc.), the non uniformity of demands (the traffic is heavier aroung railway stations or touristic sites,  for instance) which entails a higher density of antennas,  ruin any kind of invariance a statistician  could hope for. That means, we should restrict our expectations to local models of  the size of a district or a bit more. Since interference, which are the main annoyance to be dealt with, are a local phenomenon, working on a partial part of the whole domain is sufficient to predict the behavior and dimension a wireless network.

In the following sections, we will use Python code that assumes that the following packages have been loaded:

```{code-cell} ipython3
:tags: [hide-output, hide-input]

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import bernoulli
from scipy.spatial import Voronoi, ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns
font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 11,}
```

## Statistical approach
Given a circular domain with $N$ points, we want to decide whether the points exhibit repulsion or not. To do so, we will begin with a statistical approach, where we will first calculate, for Poisson processes as well as for Ginibres and $\beta$-Ginibres processes, the probability that the ratio $R = \frac{4 \pi S}{P^2}$ of the central cell is less than or equal to $r$, for values of $r$ ranging from $0$ to $1$. And then we will apply the same approach using the mean ratio of the five central cells. Finally, we will calculate $95$% confidence intervals for each of these processes.

The following code illustrates the generation of various point samples and the calculation of ratios by defining the number of points $N$ and the parameter $\beta$ for $\beta$-Ginibre processes.

```{code-cell} ipython3
:tags: [show-output, hide-input]

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
    vor.filtered_points = points_center
    vor.filtered_regions = [vor.regions[vor.point_region[i]] for i in range(len(points_center))]
    return vor

def central_area_perim(vor):  
    return ConvexHull(vor.vertices[vor.filtered_regions[0], :]).volume, ConvexHull(vor.vertices[vor.filtered_regions[0], :]).area

def area_perim(vor):
    area, perimeter = [], []
    for i in range(5):
        if len(vor.filtered_regions) >= i:
            area.append(ConvexHull(vor.vertices[vor.filtered_regions[i], :]).volume)
            perimeter.append(ConvexHull(vor.vertices[vor.filtered_regions[i], :]).area)
        else:
            area.append(np.mean(area))
            perimeter.append(np.mean(perimeter))
    return area, perimeter

def ginibre(N, cells):
    radius = (np.sqrt(N)) ; precision = 2**-53 ; error = False ; quiet=True ; output=None 
    args = [radius, N, kernels['ginibre'], precision, error, quiet, output]
    
    sample_ginibre = sample(*args)
    X_ginibre, Y_ginibre = sample_ginibre.real, sample_ginibre.imag
    
    ginibre_points = np.array([X_ginibre, Y_ginibre]).T
    indices = np.argsort((ginibre_points[:,0])**2 + ((ginibre_points[:,1])**2))
    ginibre_points = ginibre_points[indices]
    ginibre_vor = voronoi(ginibre_points, (-np.sqrt(N)-.1, np.sqrt(N)+.1, -np.sqrt(N)-.1, np.sqrt(N)+.1), len(ginibre_points))
    
    if cells==1:
        vor_area, vor_perim = central_area_perim(ginibre_vor)
    else:
        vor_area, vor_perim = area_perim(ginibre_vor)
    
    return vor_area, vor_perim

def beta_ginibre(N, beta, cells):
    radius = (np.sqrt(N)) ; precision = 2**-53 ; error = False ; quiet=True ; output=None 
    args = [radius, N, kernels['ginibre'], precision, error, quiet, output]
    
    sample_init = sample(*args)
    sample_beta_ginibre = sample_init*(bernoulli.rvs(beta, size=N))
    sample_beta_ginibre = np.array([a for a in sample_beta_ginibre if a != 0])*(np.sqrt(beta))
    X_beta_ginibre, Y_beta_ginibre = sample_beta_ginibre.real, sample_beta_ginibre.imag
    
    beta_ginibre_points = np.array([X_beta_ginibre, Y_beta_ginibre]).T
    indices = np.argsort((beta_ginibre_points[:,0])**2 + ((beta_ginibre_points[:,1])**2))
    beta_ginibre_points = beta_ginibre_points[indices]
    
    beta_ginibre_vor = voronoi(beta_ginibre_points, 
                               (-np.sqrt(N*beta)-.1, np.sqrt(N*beta)+.1, -np.sqrt(N*beta)-.1, np.sqrt(N*beta)+.1), 
                               len(beta_ginibre_points))
    
    if cells==1:
        vor_area, vor_perim = central_area_perim(beta_ginibre_vor)
    else:
        vor_area, vor_perim = area_perim(beta_ginibre_vor)
    
    return vor_area, vor_perim

def poisson(N, cells):
    radius = np.sqrt(N)
    alpha = 2 * np.pi * scipy.stats.uniform.rvs(0,1,N)
    r = radius * np.sqrt(scipy.stats.uniform.rvs(0,1,N))
    
    X_poisson, Y_poisson = r*np.cos(alpha), r*np.sin(alpha)
    poisson_points = np.array([X_poisson, Y_poisson]).T
    
    indices = np.argsort((poisson_points[:,0])**2 + ((poisson_points[:,1])**2))
    poisson_points = poisson_points[indices]
    poisson_vor = voronoi(poisson_points, (-radius -.1, radius +.1, -radius -.1, radius +.1), len(poisson_points))
    
    if cells==1:
        vor_area, vor_perim = central_area_perim(poisson_vor)
    else:
        vor_area, vor_perim = area_perim(poisson_vor)
        
    return vor_area, vor_perim

def ratio_ginibre(N, cells):
    G = ginibre(N, cells)
    return np.mean(4*np.pi*np.array(G)[0]/(np.array(G)[1])**2)

def ratio_beta_ginibre(N, beta, cells):
    beta_G = beta_ginibre(N, beta, cells)
    return np.mean(4*np.pi*np.array(beta_G)[0]/(np.array(beta_G)[1])**2)

def ratio_poisson(N, cells):
    P = poisson(N, cells)
    return np.mean(4*np.pi*np.array(P)[0]/(np.array(P)[1])**2)

%run -i Moroz_dpp.py
```

The simulation algorithm, as presented in Figure ..., provides a method for computing the quantity $\mathbb{P} \left( \frac{4 \pi S}{P^2} \le r \right)$ as a function of $r$ for the Ginibres processes (the same algorithm is applied to other processes as well). The Algorithm takes as input the number of points $N$, the number of experiences for the simulation $N_{exp}$ and the range of the varibale $r$ as a list of values. Since the simulations require a lot of time to run, we are not going to attach the associated python code, the latter is based on the algorithm described previously.

Figure {numref}`simulation-fig` shows the results of the simulations, where we compare the confidence intervals of the poisson process with the Ginibre process and the $0.7$-Ginibre process, using first the central cell and then the five central cells.

```{figure} /simulation.png
---
name: simulation-fig
---
Simulation results using the central cell (up) and the five central cells (down).
```
The limitation of the statistical approach using only the central cell is the presence of some overlap between the confidence intervals of the Poisson process and the $0.7$-Ginibre process. Consequently, in specific cases, it may not be possible to determine the true nature of some processes based on the previous statistical test. However, we can notice that using the five central cells, there is no overlap among the various curves. This is a result of averaging the ratios of the first five central cells instead of restricting the analysis to the first cell alone, a decision that provides more insights about the circular behavior of the cells for each process.

This approach shows that the chosen ratio variable represents a good repulsion criterion. On the other hand, our objective is to decide for a single map which model is the most pertinent, that is why we cannot use this approach and we will use a Machine Learning approach instead.


## Machine Learning approach
In this approach, we will use the same circular domain with $N$ points as in the statistical approach. Since the repulsion is not sensitive to scaling, we normalize the radius to $R=\sqrt{N}$. This is due to the fact that a cloud drawn from a Ginibre point process of intensity $1$ with $N$ points occupies roughly a disk with this radius. We begin by generating the data of the Ginibre process, the $0.7$-Ginibre process and the poisson process on which we will train the classification model, which is Logistic Regression Classifier. Using only the central cell (respectively the five central cells), the initial variables in our database consist of the surface and perimeter of the central cell (respectively surfaces and perimeters of the five central cells) of each generated sample, along with a binary variable that takes the value $1$ if the process is repulsive and $0$ otherwise. Subsequently, we add the ratio variable $\frac{4 \pi S}{P^2}$ of the central cell (respectively the five ratios of the five central celss) to provide the classification model with additional information on which to base its predictions.

```{code-cell} ipython3
:tags: [show-output, hide-input]

def dataframe_1cell(N, observations):
    list_df = []
    for i in range(observations):
        list_df.append(list(beta_ginibre(N, 0.7, cells=1)) + [1])
        list_df.append(list(poisson(N, cells=1)) + [0])
    df = pd.DataFrame(list_df, columns = ['A1', 'P1', 'process'])
    return df

def data_1cell(N, observations):
    list_df = []
    for i in range(observations):
        list_df.append(list(ginibre(N, cells=1)) + [1])
        list_df.append(list(poisson(N, cells=1)) + [0])
    df = pd.DataFrame(list_df, columns = ['A1', 'P1', 'process'])
    return df

def dataframe_5cells(N, observations):
    list_df = []
    for i in range(observations):
        list_df.append(sum(list(beta_ginibre(N, 0.7, cells=5)), []) + [1])
        list_df.append(sum(list(poisson(N, cells=5)), []) + [0])
    df = pd.DataFrame(list_df, columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'P1', 'P2', 'P3', 'P4', 'P5', 'process'])
    return df

def data_5cells(N, observations):
    list_df = []
    for i in range(observations):
        list_df.append(sum(list(ginibre(N, cells=5)), []) + [1])
        list_df.append(sum(list(poisson(N, cells=5)), []) + [0])
    df = pd.DataFrame(list_df, columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'P1', 'P2', 'P3', 'P4', 'P5', 'process'])
    return df
```

Here is an example of the data created with $5$ cells. We generate datas of $N_{exp} = 4000$ ($2000$ repulsive and $2000$ non-repulsive) observations of $N = 50$ points.

```{code-cell} ipython3
:tags: [show-output, show-input]

N = 100
observations = 1000

df_1cell = dataframe_1cell(N, observations)
df_1cell['R1'] = list(4*np.pi*df_1cell.A1/(df_1cell.P1)**2)
df_1cell = df_1cell[['A1', 'P1', 'R1', 'process']]

ddf_1cell = data_1cell(N, observations)
ddf_1cell['R1'] = list(4*np.pi*ddf_1cell.A1/(ddf_1cell.P1)**2)
ddf_1cell = ddf_1cell[['A1', 'P1', 'R1', 'process']]

df_5cells = dataframe_5cells(N, observations)
df_5cells['R1'] = list(4*np.pi*df_5cells.A1/(df_5cells.P1)**2)
df_5cells['R2'] = list(4*np.pi*df_5cells.A2/(df_5cells.P2)**2)
df_5cells['R3'] = list(4*np.pi*df_5cells.A3/(df_5cells.P3)**2)
df_5cells['R4'] = list(4*np.pi*df_5cells.A4/(df_5cells.P4)**2)
df_5cells['R5'] = list(4*np.pi*df_5cells.A5/(df_5cells.P5)**2)
df_5cells = df_5cells[['A1', 'P1', 'R1', 'A2', 'P2', 'R2', 'A3', 'P3', 'R3', 'A4', 'P4', 'R4', 'A5', 'P5', 'R5', 'process']]

ddf_5cells = data_5cells(N, observations)
ddf_5cells['R1'] = list(4*np.pi*ddf_5cells.A1/(ddf_5cells.P1)**2)
ddf_5cells['R2'] = list(4*np.pi*ddf_5cells.A2/(ddf_5cells.P2)**2)
ddf_5cells['R3'] = list(4*np.pi*ddf_5cells.A3/(ddf_5cells.P3)**2)
ddf_5cells['R4'] = list(4*np.pi*ddf_5cells.A4/(ddf_5cells.P4)**2)
ddf_5cells['R5'] = list(4*np.pi*ddf_5cells.A5/(ddf_5cells.P5)**2)
ddf_5cells = ddf_5cells[['A1', 'P1', 'R1', 'A2', 'P2', 'R2', 'A3', 'P3', 'R3', 'A4', 'P4', 'R4', 'A5', 'P5', 'R5', 'process']]

ddf_5cells.head()
```

## Conclusion

In this paper it has been shown numerically (based on the theoretical results in {cite}`goldman_palm_2010`) that Voronoi cells represent an effective means for determining the nature of repulsion of a configuration (repulsive or not), and this by creating a database of various configurations and extracting the areas and perimeters of the Voronoi cells in order to use them as input to the classification models described earlier.

Once the models are trained and tested on the data created, they are tested after that on real data, which are the positions of a mobile phone base stations in PARIS. Visually, we can easily say that these configurations are repulsive, which we have confirmed for the majority of these configurations by testing them by the previously trained models.

## Bibliography


```{bibliography}
:style: unsrt
```
