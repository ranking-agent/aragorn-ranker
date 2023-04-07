from ranker.shared.ranker_obj import path_collapse
import numpy as np
from pytest import approx

def test__one_hop():
    probes = [(0,1)]
    W = [[0, .5],
         [0, 0]]
    weighted_graph = np.array(W)
    measurements = [path_collapse(weighted_graph, probe) for probe in probes]
    score = np.mean(measurements)
    assert score == .5

def test_two_hop():
    probes = [(0,2)]
    W = [[0, .5, 0],
         [0, 0, .5],
         [0, 0, 0]]
    weighted_graph = np.array(W)
    measurements = [path_collapse(weighted_graph, probe) for probe in probes]
    score = np.mean(measurements)
    assert score == .25

def test_triangle():
    probes = [(0,1), (0,2), (1,2)]
    W = [[0, .5, .5],
         [0, 0, .5],
         [0, 0, 0]]
    weighted_graph = np.array(W)
    measurements = [path_collapse(weighted_graph, probe) for probe in probes]
    score = np.mean(measurements)
    assert score == .625

def test_square():
    probes = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    W = [[0, .5, 0, .5],
         [0, 0, .5, 0],
         [0, 0, 0, .5],
         [0, 0, 0, 0]]
    weighted_graph = np.array(W)
    measurements = [path_collapse(weighted_graph, probe) for probe in probes]
    score = np.mean(measurements)
    assert score == approx(.52083333333)

def test_Y_shaped():
    probes = [(0,2), (0,3), (2,3)]
    W = [[0, .5, 0, 0],
         [0, 0, .5, .5],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    weighted_graph = np.array(W)
    measurements = [path_collapse(weighted_graph, probe) for probe in probes]
    score = np.mean(measurements)
    assert score == approx(.25)

def test_triangle_with_leaf():
    probes = [(0,2), (0,3), (2,3)]
    W = [[0, .5, 0, 0],
         [0, 0, .5, .5],
         [0, 0, 0, .5],
         [0, 0, 0, 0]]
    weighted_graph = np.array(W)
    measurements = [path_collapse(weighted_graph, probe) for probe in probes]
    score = np.mean(measurements)
    assert score == approx(.41666666)