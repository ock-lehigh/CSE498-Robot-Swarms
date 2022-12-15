import numpy as np
from math import *
import matplotlib.pyplot as plt

# np.random.seed(1)

N = 10

E1 = np.mat([[0, 1], [1, 0]])
E2 = np.mat([[1, 0], [0, -1]])
E3 = np.mat([[0, -1], [1, 0]])
I = np.mat([[1, 0], [0, 1]])


q = np.mat(np.random.random(size=(N, 2)))


def calculate_μ(q, N):
    μ = np.mat(np.zeros(2))
    for i in range(N):
        μ = μ + q[i]
    return μ / N

def calculate_θ(q, μ, N):
    tx = 0
    ty = 0
    for i in range(N):
        tx = tx + (q[i] - μ) * E1 * np.transpose(q[i] - μ)
        ty = ty + (q[i] - μ) * E2 * np.transpose(q[i] - μ)
    return 0.5 * atan2(tx, ty)

def calculate_s1_s2(q, μ, N, H1, H2):
    s1 = 0
    s2 = 0
    for i in range(N):
        s1 = s1 + (q[i] - μ) * H1 * np.transpose(q[i] - μ)
        s2 = s2 + (q[i] - μ) * H2 * np.transpose(q[i] - μ)
    return s1/(2*(N-1)), s2/(2*(N-1))

def update(q, N):
    μ = calculate_μ(q, N)
    θ = calculate_θ(q, μ, N)

    R = np.mat([[cos(θ), -sin(θ)] ,[sin(θ), cos(θ)]])

    H1 = I + R * R * E2
    H2 = I - R * R * E2
    H3 = R * R * E1  

    s1, s2 = calculate_s1_s2(q, μ, N, H1, H2)

    return μ, θ, R, H1, H2, H3, s1, s2


μ, θ, R, H1, H2, H3, s1, s2 = update(q, N)


def generate_plot(q, N, μ, R, s1, s2):
    a = sqrt((N-1)*s1)
    b = sqrt((N-1)*s2)

    rect_list = []
    offset = R * np.transpose(np.mat([a,b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])
    offset = R * np.transpose(np.mat([a,-b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])
    offset = R * np.transpose(np.mat([-a,-b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])
    offset = R * np.transpose(np.mat([-a,b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])
    offset = R * np.transpose(np.mat([a,b]))
    rect_list.append([μ[0,0]+offset[0,0], μ[0,1]+offset[1,0]])

    rect = np.array(rect_list)

    xs, ys = zip(*rect)

    # plt.clf()
    plt.xlim(-1,8)
    plt.ylim(-1,8)
    plt.grid(True)
    plt.plot(xs,ys)
    for i in range(N):
        plt.plot(q[i,0], q[i,1], 'bo')
    plot_obstacle()
    # plt.pause(0.1)


def plot_obstacle():
    obstacle = np.array([[2,3], [5,2], [5,4], [2,3]])
    xs, ys = zip(*obstacle)
    plt.plot(xs,ys,'r')


plt.figure(figsize=(12,12))
plt.ion()

μd = np.mat([[0,3]])
θd = 0
s1d = 0.02
s2d = 0.05

kμ = np.mat([[1, 0], [0, 1]])
kθ = 0.5
ks1 = 0.1
ks2 = 0.1

tf = 5.
Δt = 0.1
time = np.linspace(0., tf, int(tf / Δt) + 1)

for t in time:
    uμ = (μd - μ) * kμ
    uθ = kθ * (θd - θ)
    us1 = ks1 * (s1d - s1)
    us2 = ks2 * (s2d - s2)
    uq = np.mat(np.random.random(size=(N, 2)))

    for i in range(N):
        uq[i] = uμ + np.dot(uθ*(s1-s2)/(s1+s2), np.transpose(H3*np.transpose((q[i]-μ)))) \
         + np.dot(us1/(4*s1), np.transpose(H1*np.transpose(q[i]-μ))) + np.dot(us2/(4*s2), np.transpose(H2*np.transpose(q[i]-μ)))
        q[i] += uq[i]*Δt

    μ, θ, R, H1, H2, H3, s1, s2 = update(q, N)
    plt.clf()
    generate_plot(q, N, μ, R, s1, s2)
    plt.pause(0.1)
    

def seperate(q, N, μ):
    q1_list = []
    q2_list = []
    for i in range(N):
        if q[i,0] < μ[0,0]:
            q1_list.append([q[i,0], q[i,1]])
        else :
            q2_list.append([q[i,0], q[i,1]])
    return np.mat(q1_list), len(q1_list), np.mat(q2_list), len(q2_list)
    
q1, N1, q2, N2 = seperate(q, N, μ)

μ1, θ1, R1, H11, H12, H13, s11, s12 = update(q1, N1)
μ2, θ2, R2, H21, H22, H23, s21, s22 = update(q2, N2)

for t in range(5):
    plt.clf()
    generate_plot(q1, N1, μ1, R1, s11, s12)
    generate_plot(q2, N2, μ2, R2, s21, s22)
    plt.pause(0.1)

for t in range(10):
    for i in range(N1):
        q1[i,0] -= 0.03
    μ1, θ1, R1, H11, H12, H13, s11, s12 = update(q1, N1)

    for i in range(N2):
        q2[i,0] += 0.05
    μ2, θ2, R2, H21, H22, H23, s21, s22 = update(q2, N2)

    plt.clf()
    generate_plot(q1, N1, μ1, R1, s11, s12)
    generate_plot(q2, N2, μ2, R2, s21, s22)
    plt.pause(0.1)

μ1d = np.mat([[0.1,4]])
θ1d = 0
s11d = 0.02
s12d = 0.02

kμ1 = np.mat([[1, 0], [0, 1]])
kθ1 = 0.5
ks11 = 0.1
ks12 = 0.1

μ2d = np.mat([[-0.1,2]])
θ2d = 0
s21d = 0.02
s22d = 0.02

kμ2 = np.mat([[1, 0], [0, 1]])
kθ2 = 0.5
ks21 = 0.1
ks22 = 0.1

tf = 3.
Δt = 0.1
time = np.linspace(0., tf, int(tf / Δt) + 1)

for t in time:
    uμ1 = (μ1d - μ1) * kμ1
    uθ1 = kθ1 * (θ1d - θ1)
    us11 = ks11 * (s11d - s11)
    us12 = ks12 * (s12d - s12)
    uq1 = np.mat(np.random.random(size=(N1, 2)))

    for i in range(N1):
        uq1[i] = uμ1 + np.dot(uθ1*(s11-s12)/(s11+s12), np.transpose(H13*np.transpose((q1[i]-μ1)))) \
         + np.dot(us11/(4*s11), np.transpose(H11*np.transpose(q1[i]-μ1))) + np.dot(us12/(4*s12), np.transpose(H12*np.transpose(q1[i]-μ1)))
        q1[i] += uq1[i]*Δt

    μ1, θ1, R1, H11, H12, H13, s11, s12 = update(q1, N1)

    uμ2 = (μ2d - μ2) * kμ2
    uθ2 = kθ2 * (θ2d - θ2)
    us21 = ks21 * (s21d - s21)
    us22 = ks22 * (s22d - s22)
    uq2 = np.mat(np.random.random(size=(N2, 2)))

    for i in range(N2):
        uq2[i] = uμ2 + np.dot(uθ2*(s21-s22)/(s21+s22), np.transpose(H23*np.transpose((q2[i]-μ2)))) \
         + np.dot(us21/(4*s21), np.transpose(H21*np.transpose(q2[i]-μ2))) + np.dot(us22/(4*s22), np.transpose(H22*np.transpose(q2[i]-μ2)))
        q2[i] += uq2[i]*Δt

    μ2, θ2, R2, H21, H22, H23, s21, s22 = update(q2, N2)

    plt.clf()
    generate_plot(q1, N1, μ1, R1, s11, s12)
    generate_plot(q2, N2, μ2, R2, s21, s22)
    plt.pause(0.1)
    
        
μ1d = np.mat([[7,6]])
θ1d = 0
s11d = 0.02
s12d = 0.02

kμ1 = np.mat([[0.7, 0], [0, 0.7]])
kθ1 = 0.5
ks11 = 0.1
ks12 = 0.1

μ2d = np.mat([[7,0]])
θ2d = 0
s21d = 0.02
s22d = 0.02

kμ2 = np.mat([[0.7, 0], [0, 0.7]])
kθ2 = 0.5
ks21 = 0.1
ks22 = 0.1

tf = 5.
Δt = 0.1
time = np.linspace(0., tf, int(tf / Δt) + 1)

for t in time:
    uμ1 = (μ1d - μ1) * kμ1
    uθ1 = kθ1 * (θ1d - θ1)
    us11 = ks11 * (s11d - s11)
    us12 = ks12 * (s12d - s12)
    uq1 = np.mat(np.random.random(size=(N1, 2)))

    for i in range(N1):
        uq1[i] = uμ1 + np.dot(uθ1*(s11-s12)/(s11+s12), np.transpose(H13*np.transpose((q1[i]-μ1)))) \
         + np.dot(us11/(4*s11), np.transpose(H11*np.transpose(q1[i]-μ1))) + np.dot(us12/(4*s12), np.transpose(H12*np.transpose(q1[i]-μ1)))
        q1[i] += uq1[i]*Δt

    μ1, θ1, R1, H11, H12, H13, s11, s12 = update(q1, N1)

    uμ2 = (μ2d - μ2) * kμ2
    uθ2 = kθ2 * (θ2d - θ2)
    us21 = ks21 * (s21d - s21)
    us22 = ks22 * (s22d - s22)
    uq2 = np.mat(np.random.random(size=(N2, 2)))

    for i in range(N2):
        uq2[i] = uμ2 + np.dot(uθ2*(s21-s22)/(s21+s22), np.transpose(H23*np.transpose((q2[i]-μ2)))) \
         + np.dot(us21/(4*s21), np.transpose(H21*np.transpose(q2[i]-μ2))) + np.dot(us22/(4*s22), np.transpose(H22*np.transpose(q2[i]-μ2)))
        q2[i] += uq2[i]*Δt

    μ2, θ2, R2, H21, H22, H23, s21, s22 = update(q2, N2)

    plt.clf()
    generate_plot(q1, N1, μ1, R1, s11, s12)
    generate_plot(q2, N2, μ2, R2, s21, s22)
    plt.pause(0.1)


μ1d = np.mat([[7,3.25]])
θ1d = 0
s11d = 0.02
s12d = 0.02

kμ1 = np.mat([[1, 0], [0, 1]])
kθ1 = 0.5
ks11 = 0.1
ks12 = 0.1

μ2d = np.mat([[7,2.75]])
θ2d = 0
s21d = 0.02
s22d = 0.02

kμ2 = np.mat([[1, 0], [0, 1]])
kθ2 = 0.5
ks21 = 0.1
ks22 = 0.1

tf = 4.
Δt = 0.1
time = np.linspace(0., tf, int(tf / Δt) + 1)

for t in time:
    uμ1 = (μ1d - μ1) * kμ1
    uθ1 = kθ1 * (θ1d - θ1)
    us11 = ks11 * (s11d - s11)
    us12 = ks12 * (s12d - s12)
    uq1 = np.mat(np.random.random(size=(N1, 2)))

    for i in range(N1):
        uq1[i] = uμ1 + np.dot(uθ1*(s11-s12)/(s11+s12), np.transpose(H13*np.transpose((q1[i]-μ1)))) \
         + np.dot(us11/(4*s11), np.transpose(H11*np.transpose(q1[i]-μ1))) + np.dot(us12/(4*s12), np.transpose(H12*np.transpose(q1[i]-μ1)))
        q1[i] += uq1[i]*Δt

    μ1, θ1, R1, H11, H12, H13, s11, s12 = update(q1, N1)

    uμ2 = (μ2d - μ2) * kμ2
    uθ2 = kθ2 * (θ2d - θ2)
    us21 = ks21 * (s21d - s21)
    us22 = ks22 * (s22d - s22)
    uq2 = np.mat(np.random.random(size=(N2, 2)))

    for i in range(N2):
        uq2[i] = uμ2 + np.dot(uθ2*(s21-s22)/(s21+s22), np.transpose(H23*np.transpose((q2[i]-μ2)))) \
         + np.dot(us21/(4*s21), np.transpose(H21*np.transpose(q2[i]-μ2))) + np.dot(us22/(4*s22), np.transpose(H22*np.transpose(q2[i]-μ2)))
        q2[i] += uq2[i]*Δt

    μ2, θ2, R2, H21, H22, H23, s21, s22 = update(q2, N2)

    plt.clf()
    generate_plot(q1, N1, μ1, R1, s11, s12)
    generate_plot(q2, N2, μ2, R2, s21, s22)
    plt.pause(0.1)

def reunit(q1, N1, q2, N2):
    q_list = []
    for i in range(N1):
        q_list.append([q1[i,0], q1[i,1]])
    for i in range(N2):
        q_list.append([q2[i,0], q2[i,1]])
    return np.mat(q_list), len(q_list)
    
q, N = reunit(q1, N1, q2, N2)

μ, θ, R, H1, H2, H3, s1, s2 = update(q, N)

for t in range(5):
    plt.clf()
    generate_plot(q, N, μ, R, s1, s2)
    plt.pause(0.1)


μd = np.mat([[7,3]])
θd = 0
s1d = 0.1
s2d = 0.1

kμ = np.mat([[1, 0], [0, 1]])
kθ = 0.5
ks1 = 0.1
ks2 = 0.1

tf = 3.
Δt = 0.1
time = np.linspace(0., tf, int(tf / Δt) + 1)

for t in time:
    uμ = (μd - μ) * kμ
    uθ = kθ * (θd - θ)
    us1 = ks1 * (s1d - s1)
    us2 = ks2 * (s2d - s2)
    uq = np.mat(np.random.random(size=(N, 2)))

    for i in range(N):
        uq[i] = uμ + np.dot(uθ*(s1-s2)/(s1+s2), np.transpose(H3*np.transpose((q[i]-μ)))) \
         + np.dot(us1/(4*s1), np.transpose(H1*np.transpose(q[i]-μ))) + np.dot(us2/(4*s2), np.transpose(H2*np.transpose(q[i]-μ)))
        q[i] += uq[i]*Δt

    μ, θ, R, H1, H2, H3, s1, s2 = update(q, N)
    plt.clf()
    generate_plot(q, N, μ, R, s1, s2)
    plt.pause(0.1)


plt.ioff()
plt.show()
