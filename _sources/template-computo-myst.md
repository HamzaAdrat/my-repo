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
In the performance analysis of cellular systems, the locations of antennas (or base stations) play a major role (see {cite}`BaccelliStochasticGeometryWireless2008`). It is usually admitted that they can be modeled by a Poisson process. But the data which can be gathered from the Web site of the French National Agency of Radio Frequencies, Cartoradio, see {cite}`ANFR`, tend to prove that this may not be the case. More precisely, if we look at the global picture of all antennas in Paris, we see features reminiscent of a Poisson process (local clusters for instance), see {numref}`paris-fig`. However, if we look closer and finer, by specifying a region and a frequency band, we see that the antennas locations do seem to exhibit some repulsion (see {numref}`orange-fig`).

```{figure} my-repo/paris.pdf
---
name: paris-fig
---
Antennas in Paris
```

```{figure} my-repo/orange.pdf
---
name: orange-fig
---
Antennas in one frequency  band only
```
In previous papers, point processes with repulsion have been used to model such systems . The question is then to decide, given one sample of positions of base stations in a bounded domain, whether it is more likely to be modeled by a point process with repulsion or by a *neutral* point process, i.e. where the locations are independent. As we only have a single realization, we cannot use frequency methods. Since the observation window is finite, we cannot either resort to estimates based on stationarity or ergodicity and we must take care from the side effects.

In previous papers, point processes with repulsion have been used to model such systems {cite}`Deng2014`, {cite}`Miyoshi2016`, {cite}`gomez_case_2015` for no reason but a mere resemblance between the pictures like {numref}`orange-fig` and those obtained by simulating a point process with repulsion. The question is then to decide, given one sample of positions of base stations in a bounded domain, whether it is more likely to be modeled by a point process with repulsion or by a *neutral* point process, i.e. where the locations could be considered as coming from independent drawings of some identically distributed random variables. As we only have a single realization,  we cannot use frequency methods. Since the observation window is finite, we cannot either resort to estimates based on stationarity or ergodicity and  we must take care from the side effects.

The rationale behind our work comes from {cite}`goldman_palm_2010`. It is shown there  that the Voronoi cells of the Ginibre point process (a particular point process with repulsion, see below for the exact definition) are in some sense more regular (closer to a circle) than those of a Poisson process (see Theorem later!). By simulation, this feature seem to persist for other point processes with repulsion, like Gibbs processes. It is this aspect that we use to construct our classification algorithm.
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

## Methods


(subsec:this-is-a-subheading)=
### This is a subheading

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur eget ipsum
est. Mauris purus urna, aliquet non interdum quis, tincidunt in tortor.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac
turpis egestas. Vestibulum efficitur eu mi ac consequat. Sed id velit porta
libero facilisis auctor malesuada ut erat. Sed at orci sem. Nunc eget nisi
lobortis, ultrices justo vel, porttitor dolor. Proin ullamcorper dui purus, eu
maximus lorem pharetra non. Etiam ante purus, tempus eget rhoncus eu, rutrum
id ante. Nulla suscipit, risus vel dapibus finibus, metus diam euismod felis,
sit amet consequat quam lectus quis lacus. Sed tristique, urna sit amet
viverra facilisis, est augue auctor lorem, sed varius lectus nibh eget urna.
Aenean a luctus ligula, vitae elementum metus. Etiam varius, leo in iaculis
rutrum, mauris ex fringilla enim, vitae iaculis justo massa id purus. 

```{code-cell} python3
---
tags: [show-output, show-input]
---
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(np.arange(10))
```

FIXME: Can we reference figures?

(subsec:subheading)=
### This is another subheading

As seen in [section](subsec:this-is-a-subheading), lorem ipsum dolor sit amet,
consectetur adipiscing elit. Cras nec ornare urna. Nullam consectetur
vestibulum lacinia. Aenean sed augue elit. Aenean venenatis orci ut felis
condimentum, sit amet feugiat eros tincidunt. Vestibulum ante ipsum primis in
faucibus orci luctus et ultrices posuere cubilia curae; Integer varius metus
nunc, at molestie metus eleifend et. Maecenas ullamcorper metus at nisl
molestie, ac commodo arcu mollis. Donec felis odio, fermentum lacinia
vestibulum non, elementum eu metus. Donec suscipit aliquam malesuada. Praesent
justo turpis, dignissim ac nulla non, malesuada rutrum nisi.

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

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent aliquam
porttitor rutrum. Donec in sollicitudin risus, ultrices ultricies nisi.
Vestibulum vel turpis eros. Suspendisse pellentesque nunc augue, eget laoreet
dui dictum ut. Mauris pretium lectus ut elit pulvinar, nec accumsan purus
hendrerit. Phasellus hendrerit orci a vestibulum euismod. Nunc vel massa
facilisis, cursus justo nec, suscipit est. 

- This is a list
- With more elements
- It isn't numbered.

But we can also do a numbered list

1. This is my first item
2. This is my second item
3. This is my third item

## Conclusion

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec molestie mollis
urna, vitae convallis turpis placerat vel. Orci varius natoque penatibus et
magnis dis parturient montes, nascetur ridiculus mus. Vestibulum elit nulla,
laoreet a sagittis non, pharetra ac velit. Aliquam volutpat nisl augue, eget
maximus erat cursus eget. Maecenas sed nisi bibendum, sagittis tellus et,
lacinia augue. In lobortis, libero at auctor aliquam, mi leo egestas orci, a
pulvinar mauris magna in nisi. Morbi cursus dictum dignissim. Fusce at ex sit
amet felis vehicula gravida non in sem. Morbi ut condimentum diam. Aliquam
erat volutpat. Fusce id pharetra ante, tincidunt dapibus eros. Curabitur
mattis magna non felis aliquet sagittis. 

```{bibliography}
:style: unsrt
```
