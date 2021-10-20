/* MPI Program Template */
// Heteregenous MPI+X implementation of classification 

#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono> 
#include <omp.h>
#include "mpi.h"
#include "mm.h"
using namespace std; 

#define ll long long int
#define umap unordered_map
#define pb push_back
#define ff first
#define ss second
#define sz(a) (int)a.size()
#define all(X) X.begin(),X.end()
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

int N; // total no. of instances
int d_x; // no. of dimensions in data vector
int d_y; // no. of possible labels
const int k = 10; // branching factor

const int level_threshold = 800;
const int n_leaf = 10;

vector< unordered_map<int,double> > X; // Feature Matrix
vector< unordered_map<int,double> > Y; // Label matrix
vector< unordered_map<int,double> > Xp; // Projected feature Matrix
vector< unordered_map<int,double> > Yp; // Projected label matrix
vector< unordered_map<int,double> > mean; // mean vector for storing meanLabels at leaves
vector< vector<int> > child_set; // set of children ids for a given node
vector<int> pos; // reference to positions for a node id in mean vector and classifier vector
vector< vector< unordered_map<int,double> > > classifier; // stores the classifiers for each node
vector<double> Mg;
vector<bool> leaf_mrk;

int Nt;

string train_file,test_file;

vector< unordered_map<int,double> > Xt; // Feature Matrix
vector< unordered_map<int,double> > Yt; // Label matrix
vector< unordered_map<int,double> > Xpt; // Projected feature Matrix
vector< unordered_map<int,double> > Ypt; // Projected label matrix
vector<double> Mgt;
vector<double> meanvector,globalmeanvector;
vector<pair<double,int> > cmpvector;
vector<int> bel;

vector<vector<int> > prx;
vector<vector<double> > pry;

/* -------------------------------------------------------*/

int xdim,ydim,projDimx,projDimy;
ll seedIndexX,seedSignX,seedIndexY,seedSignY;
vector< vector<int> > ProjColX,ProjSignX,ProjColY,ProjSignY;

int xdimt,ydimt,projDimxt,projDimyt;
ll seedIndexXt,seedSignXt,seedIndexYt,seedSignYt;
vector<int> ProjColXt,ProjSignXt,ProjColYt,ProjSignYt;

// input processing and projection

void getInpX()
{
    ifstream in(train_file.c_str());
    string s;
    getline(in,s);
    int c = 0;
    while(c<sz(s))
    {
        if(s[c]>='0' && s[c]<='9')
        {
            N = (N*10+(s[c]-'0'));
            c++;
            continue;
        }
        c++;
        break;
    }
    while(c<sz(s))
    {
        if(s[c]>='0' && s[c]<='9')
        {
            xdim = (xdim*10+(s[c]-'0'));
            c++;
            continue;
        }
        c++;
        break;
    }
    while(c<sz(s))
    {
        if(s[c]>='0' && s[c]<='9')
        {
            ydim = (ydim*10+(s[c]-'0'));
            c++;
            continue;
        }
        c++;
        break;
    }
    int row = 0;
    while(getline(in,s))
    {
        umap<int,double> vec;
        int c = 0;
        while(1)
        {
            int v = 0;
            while(c<sz(s))
            {
                if(s[c]>='0' && s[c]<='9')
                {
                    v = (v*10+(s[c]-'0'));
                    c++;
                    continue;
                }
                break;
            }
            vec[v] = 1;
            if(s[c] == ' ')
                break;
            c++;
        }
        Y.pb(vec);
        
        vec.clear();
        c++;
        while(c<s.size())
        {
            int col = 0;
            while(s[c]!=':')
            {
                col = (col*10+s[c]-'0');
                c++;
            }
            c++;
            double d = 0;
            double val = 0;
            while(c<s.size() && s[c]!=' ')
            {
                if(s[c] == '.')
                {
                    d = 1;
                    c++;
                    continue;
                }
                val = (val*10+s[c]-'0');
                c++;
                d = d*10;
            }
            d = max(1.0,d);
            vec[col] = val/d;
            c++;
        }
        X.pb(vec);
    }
}

void getTstX()
{
    ifstream in(test_file.c_str());
    string s;
    getline(in,s);
    int c = 0;
    while(c<sz(s))
    {
        if(s[c]>='0' && s[c]<='9')
        {
            Nt = (Nt*10+(s[c]-'0'));
            c++;
            continue;
        }
        c++;
        break;
    }
    while(c<sz(s))
    {
        if(s[c]>='0' && s[c]<='9')
        {
            xdimt = (xdimt*10+(s[c]-'0'));
            c++;
            continue;
        }
        c++;
        break;
    }
    while(c<sz(s))
    {
        if(s[c]>='0' && s[c]<='9')
        {
            ydimt = (ydimt*10+(s[c]-'0'));
            c++;
            continue;
        }
        c++;
        break;
    }
    int row = 0;
    while(getline(in,s))
    {
        umap<int,double> vec;
        int c = 0;
        while(1)
        {
            int v = 0;
            while(c<sz(s))
            {
                if(s[c]>='0' && s[c]<='9')
                {
                    v = (v*10+(s[c]-'0'));
                    c++;
                    continue;
                }
                break;
            }
            vec[v] = 1;
            if(s[c] == ' ')
                break;
            c++;
        }
        Yt.pb(vec);
        
        vec.clear();
        c++;
        while(c<s.size())
        {
            int col = 0;
            while(s[c]!=':')
            {
                col = (col*10+s[c]-'0');
                c++;
            }
            c++;
            double d = 0;
            double val = 0;
            while(c<s.size() && s[c]!=' ')
            {
                if(s[c] == '.')
                {
                    d = 1;
                    c++;
                    continue;
                }
                val = (val*10+s[c]-'0');
                c++;
                d = d*10;
            }
            d = max(1.0,d);
            vec[col] = val/d;
            c++;
        }
        Xt.pb(vec);
    }
}

void compute_magnitudes_train()
{
    double mg = 0;
    Mg.clear();
    for(int i=0;i<N;++i)
    {
        for(auto u:Xp[i])
            mg += u.second*u.second;
        Mg.push_back(sqrt(mg));
    }
}

void compute_magnitudes_test(int i)
{
    double mg = 0;
    for(auto u:Xpt[i])
        mg += u.second*u.second;
    Mgt[i] = mg;
}

void intialize_seeds()
{
    ll mod = (1LL<<32)-1;
    srand(time(NULL));
    seedIndexX = (rand()*rand()*rand())%mod;
    seedIndexY = (rand()*rand()*rand())%mod;
    seedSignX = (rand()*rand()*rand())%mod;
    seedSignY = (rand()*rand()*rand())%mod;
}

ll getHash(int i,ll mod,ll seed)
{
    const int key = i;
    uint32_t p[4];
    MurmurHash3_x86_32(&key, sizeof(int), seed, &p);            
    ll res = (ll)p[0];
    return res%mod;
}

void ProjectX(int tree)
{
    unordered_map<int,double> tp;
    Xp.clear();
    for(int i=0;i<N;++i)
        Xp.push_back(tp);
    for(int i=0;i<N;i++)
        for(auto u:X[i])
            Xp[i][ProjColX[tree][u.ff]] += (double)ProjSignX[tree][u.ff]*u.ss;
}

void ProjectY(int tree)
{
    unordered_map<int,double> tp;
    Yp.clear();
    for(int i=0;i<N;++i)
        Yp.push_back(tp);
    for(int i=0;i<N;i++)
        for(auto u:Y[i])
            Yp[i][ProjColY[tree][u.ff]] += (double)ProjSignY[tree][u.ff]*u.ss;
}

void ProjectXt(int tree,int v)
{
    unordered_map<int,double> tp;
    Xpt[v] = tp;
    for(auto u:Xt[v])
        Xpt[v][ProjColX[tree][u.ff]] += (double)ProjSignX[tree][u.ff]*u.ss;
}

void ProjectYt(int tree,int v)
{
    unordered_map<int,double> tp;
    Ypt[v] = tp;
    for(auto u:Yt[v])
        Ypt[v][ProjColY[tree][u.ff]] += (double)ProjSignY[tree][u.ff]*u.ss;
}

void __init(int trees)
{
    getInpX();
    getTstX();
    projDimx = min(xdim,10000);
    projDimy = min(ydim,10000);

    vector<int> tp;
    for(int j=0;j<trees;++j)
    {
        ProjColX.pb(tp);
        ProjSignX.pb(tp);
        ProjColY.pb(tp);
        ProjSignY.pb(tp);
    }

    for(int j=0;j<trees;++j)
    {
        intialize_seeds();
        for(int i = 0;i < xdim;i++)
        {
            ProjColX[j].pb(getHash(i,projDimx,seedIndexX));
            ProjSignX[j].pb(2*getHash(i,2,seedSignX)-1);
        }
        for(int i = 0;i < ydim;i++)
        {
            ProjColY[j].pb(getHash(i,projDimy,seedIndexY));
            ProjSignY[j].pb(2*getHash(i,2,seedSignY)-1);
        }
    }

    bel.resize(N);
}

/* -------------------------------------------------------*/

int NXT_NODE_ID;

// check for leaf condition
bool test_leaf(vector<int> instances,int level)
{
    // max depth reached
    if(level == level_threshold)
    {
        // cout << "max level reached : " << level << endl;
        return 1;
    }
    
    // cardinality of the node’s instance subset is lower than a given threshold
    int n = instances.size();
    if(n <= n_leaf)
    {
        // cout << "less instances than threshold : " << n << endl;
        return 1;
    }

    int i,j;
    // cout << "no. of instances : " << n << endl;
    // all the instances have the same features
    for(i=1;i<n;++i)
    {
        int cur = instances[i];
        int prv = instances[i-1];
        int d = X[cur].size();
        if(d != (int) X[prv].size())
            break;
        
        bool flag = 1;
        for(auto v:X[cur])
        {
            int dim = v.first;
            double val = v.second;
            if(abs(X[prv][dim]-val) > 1e-5)
            {
                flag = 0;
                break;
            }
        }
        if(!flag)
            break;
    }
    if(i == n)
    {
        // cout << "All instances found to have same features" << endl;
        return 1;
    }

    // all the instances have the same labels
    for(i=1;i<n;++i)
    {
        int cur = instances[i];
        int prv = instances[i-1];
        int d = Y[cur].size();
        if(d != (int) Y[prv].size())
            break;
        
        bool flag = 1;
        for(auto v:Y[cur])
        {
            int dim = v.first;
            double val = v.second;
            if(abs(Y[prv][dim]-val) > 1e-5)
            {
                flag = 0;
                break;
            }
        }
        if(!flag)
            break;
    }
    if(i == n)
    {
        // cout << "All instances found to have same labels" << endl;
        return 1;
    }

    // cout << "Check : node is non leaf." << endl;
    return 0;
}

// computes meanLabel for the leaves
unordered_map<int,double> compute_meanLabel(vector<int> instances)
{
    int n = instances.size();
    unordered_map<int,double> meanLabel;
    for(int i=0;i<n;++i)
    {
        int cur = instances[i];
        for(auto v:Y[cur])
        {
            int dim = v.first;
            meanLabel[dim]+=(1.0/(double)(n));
        }
    }
    return meanLabel;
}

// Spherical K-means
// Instead of normal distance measure, using cosine similarity for getting closest cluster in K-means
vector<int> spherical_kmeans(vector<int> inds)
{
    // cout << "K-means ... " << (int)(inds.size())  << endl;
    // inds = Indices which have been chosen to build classifier at current node
    // k = number of clusters
    int n = inds.size();
    vector<umap<int,double>> centroids;
    
    // Store magnitudes of data points 
    vector<double> mags(n,0);
    for(int i=0;i<n;i++)
    {
        int id = inds[i];

        double res = 0;
        for(auto u:Yp[id])
            res = res + u.ss*u.ss;
        
        mags[i] = sqrt(res);
    }

    // Choose the first k points as centroids 
    for(int i=0;i<k;i++)
    {
        int id = inds[i];
        
        umap<int,double> vec;
        if(mags[i]==0)
        {
            vec[0] = 1;
            centroids.pb(vec);
            continue;
        }
        for(auto u:Yp[id])
        {
            vec[u.ff] = u.ss / mags[i];
        }
        centroids.pb(vec);
    }

    vector<int> labels(n,0);
    int iterations = 0;
    while(1)
    {
        ++iterations;
        // Spherical K-means loop
        vector<umap<int,double>> new_centroids;
        for(int i=0;i<k;i++)
        {
            umap<int,double> vec;
            new_centroids.pb(vec);
        }
        
        for(int i=0;i<n;i++)
        {
            int id = inds[i];
            // Find the distance of point from the centroids.
            vector<pair<double,int>> dist;
            for(int j=0;j<k;j++)
            {
                // find dot product of unit vectors of Xp[id] and centroid[j]
                double d = 0;
                for(auto u:Yp[id])
                {
                    auto it = centroids[j].find(u.ff);
                    if(it!=centroids[j].end() && mags[i])
                        d = d + (u.ss*(it->ss))/mags[i];
                }
                dist.pb({1.00-d,j});
            }
            // Assign current point to the closest cluster
            sort(all(dist));
            labels[i] = dist[0].ss;
        }

        // Find new centroids
        for(int i=0;i<n;i++)
        {
            if(mags[i]==0)
                continue;
            int id = inds[i];
            int cent = labels[i];
            for(auto u:Yp[id])
            {
                new_centroids[cent][u.ff] += u.ss/mags[i];
            }
        }
        // Normalize the new centroids.
        for(int j=0;j<k;j++)
        {
            double res = 0;
            for(auto u:new_centroids[j])
                res = res + u.ss*u.ss;
            res = sqrt(res);
            if(res==0)
            {
                res = 1;
                new_centroids[j][0] = 1;
                continue;
            }
            for(auto it = new_centroids[j].begin();it!=new_centroids[j].end();it++)
            {
                it->ss /= res;
            }
        }
        // Check for difference in previous and current centroid.
        bool f = 0;
        double eps = 1e-8;
        for(int i=0;i<k;i++)
        {
            for(auto u:centroids[i])
            {
                double df;
                auto it = new_centroids[i].find(u.ff);
                if(it==new_centroids[i].end())
                    df = u.ss;
                else
                    df = abs(u.ss - it->ss);
                if(df > eps)
                    f=1;
            }
            for(auto u:new_centroids[i])
            {
                double df;
                auto it = centroids[i].find(u.ff);
                if(it == centroids[i].end())
                    df = u.ss;
                else
                    df = abs(u.ss - it->ss);
                if(df > eps)
                    f=1;
            }
        }
        if(f==0)
            break;
        centroids = new_centroids;
    }
    // cout << "Check : clusters built on sampled labels." << endl;
    return labels;  
}

// building classifier after performing k-means on sampled data and labels
vector<umap<int,double>> buildClassifier(vector<int> labels,vector<int> inds)
{
    // Find centroids of X
    vector<umap<int,double>> centroids;
    for(int i=0;i<k;i++)
    {
        umap<int,double> vec;
        centroids.pb(vec);
    }
    int n = inds.size();
    for(int i=0;i<n;i++)
    {
        int id = inds[i];

        double mag = 0;
        for(auto u:Xp[id])
            mag = mag + u.ss*u.ss;
        if(mag==0)
            continue;
        mag = sqrt(mag);
        for(auto u:Xp[id])
            centroids[labels[i]][u.ff] += u.ss/mag;
    }

    for(int i=0;i<k;i++)
    {
        double mag = 0;
        for(auto u:centroids[i])
            mag = mag + u.ss*u.ss;
        mag = sqrt(mag);
        if(mag == 0)
        {
            centroids[i][0] = 1;
            continue;
        }
        for(auto it=centroids[i].begin();it!=centroids[i].end();it++)
            it->ss /= mag;
    }
    return centroids;
}

// training the node classifier
vector< unordered_map<int,double> > train_node_classifier(vector<int> inds)
{   
    shuffle(all(inds),rng); 
    int smp = min(sz(inds),20000);
    vector<int> samples;
    for(int i=0;i<smp;i++)
        samples.pb(inds[i]);
    // Call spherical K-means to get Labels
    vector<int> labels = spherical_kmeans(samples);
    // After getting Lables call buildClassifier to get Classifier.
    vector< unordered_map<int,double> > node_classifier = buildClassifier(labels,samples); 
    // cout << "Check : Node trained, classifier built." << endl;
    return node_classifier;
}

// 1-cosine_similarity(vector a,vector b)
// flag 0 is for training and flag 1 is for testing
// p is the pos for node and v is the index of data to be classfied in train/test matrix
// i is the classifier index for the concerned node
double similarity(int u,int p,int i,int flag)
{
    double prd = 0;
    if(flag == 0)
    {
        for(auto v:Xp[u])
            prd += v.second*classifier[p][i][v.first];
        return 1.0-(prd/Mg[u]);
    }
    else
    {
        for(auto v:Xpt[u])
            prd += v.second*classifier[p][i][v.first];
        return 1.0-(prd/Mgt[u]);
    }
}

// Given certain data points in clusters, classify function decides for test point, the most similar data point
// flag 0 is for training and flag 1 is for testing
// p is the pos for node and v is the index of data to be classfied in train/test matrix 
int classify(int p,int v,int flag)
{
    double mx = 1e12;
    int cid = 0;
    for(int i=0;i<(int)classifier[p].size();++i)
    {
        // to hold test point to be moved only into children containing data
        if(flag == 1 && classifier[p][i][0] == 1e9)
            continue;
        double sim = similarity(v,p,i,flag);
        if(sim < mx)
        {
            mx = sim;
            cid = i;
        }
    }
    // if(flag == 1)
    //  cout << "similarity : " << mx << endl;
    // cout << "Check : classified to " << cid << endl;
    return cid;
}

// build function for constructing the tree
int train_tree(vector<int> instances,int level)
{
    int id = NXT_NODE_ID;
    // cout << "processing node " << id << endl; 
    NXT_NODE_ID++;
    if(test_leaf(instances, level))
    {
        // Here pos indicates position of mean label for this node present in mean vector 
        leaf_mrk.push_back(true);
        pos.push_back((int) mean.size());
        mean.push_back(compute_meanLabel(instances));
        // cout << "found leaf node " << id << endl;
        return id;
    }

    leaf_mrk.push_back(false);
    // Here pos indicates position of classifier for this node present in classifier vector 
    pos.push_back((int) classifier.size());
    classifier.push_back(train_node_classifier(instances));
    vector<int> children(k,-1);
    child_set.push_back(children);
    vector< vector<int> > partition(k);

    // cout << "Classification begins .. " << (int)(classifier[pos[id]].size()) << "\n";
    int n = instances.size();
    int p = pos[id];
    // cout << "there" << endl;

    # pragma omp parallel for  // num_threads(min(n,1))
    for(int i=0;i<n;++i)
    {
        int v = instances[i];
        double mx = 1e6;
        int cid = 0;
        for(int j=0;j<k;++j)
        {
            // printf("i = %d, j= %d, v = %d, threadId = %d \n", i, j, v, omp_get_thread_num());
            double sim = 0.0, prd = 0.0;
            for(int l=0;l<(int)prx[v].size();++l)
            {
                auto it = classifier[p][j].find(prx[v][l]);
                if(it != classifier[p][j].end())
                    prd += (pry[v][l]/Mg[v])*it->second;
            }
            sim = 1.0-prd;
            if(sim < mx)
            {
                mx = sim;
                cid = j;
            }
        }
        bel[v] = cid;
    }


    // cout << "here" << endl;
    int mn = 10;
    int mx = -1;
    for(auto v:instances)
    {
        mn = min(mn,bel[v]);
        mx = max(mx,bel[v]);
        partition[bel[v]].push_back(v);
    }

    if(mn < 10 && mx > -1 && mn == mx)
    {
        leaf_mrk.pop_back();
        pos.pop_back();
        classifier.pop_back();
        child_set.pop_back();

        leaf_mrk.push_back(true);
        pos.push_back((int) mean.size());
        mean.push_back(compute_meanLabel(instances));
        // cout << "found leaf node " << id << endl;
        return id;
    }

    int ch = 0;
    for(int i=0;i<k;++i)
    {
        if(!(partition[i].empty()))
        {
            int rid = train_tree(partition[i],level+1);
            child_set[pos[id]][i] = rid;
            ++ch;
        }
        else
        {
            classifier[pos[id]][i][0] = 1e9; // INFINITE value set as an indicator
        }
    }
    // cout << "node " << id << " has " << ch << " children" << endl;
    // for(auto v:child_set[pos[id]])
    //  cout << v << " ";
    // cout << endl;
    return id;
}

double p[6];

// classify testdata
// id is the node in tree and v is the index of concerned testdata in the test_feature matrix
void test(int id,int v)
{
    // cout << "test " << id << " " << v << " " << leaf_mrk[id] << endl;
    if(leaf_mrk[id])
    {
        for(auto v:mean[pos[id]])
            meanvector[v.first] += v.second;   
        return;
    }
    int cid = classify(pos[id],v,1);
    // cout << "descends " << cid << " " << (int)(child_set[pos[id]].size()) << " " << child_set[pos[id]][cid] << endl;
    test(child_set[pos[id]][cid],v);
}

void copy_projections()
{
    prx.clear();
    pry.clear();
    for(int i=0;i<N;++i)
    {
        vector<int> itp;
        vector<double> dtp;
        prx.push_back(itp);
        pry.push_back(dtp);
        for(auto v:Xp[i])
        {
            prx[i].push_back(v.first);
            pry[i].push_back(v.second);
        }
    }
}

int main( int argc, char **argv ) {
    int rank, numprocs;

    /* start up MPI */
    MPI_Init( &argc, &argv );

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &numprocs );

    train_file = argv[1];
    test_file = argv[2];
    
    /*synchronize all processes*/
    MPI_Barrier( MPI_COMM_WORLD );
    double tbeg = MPI_Wtime();
    double elapsedTime = MPI_Wtime();
    double train_time = 0;

    int trees = stoi(argv[3]);
    __init(trees);
    vector<int> instances(N);
    for(int i=0;i<N;++i)
        instances[i] = i;
    tbeg = MPI_Wtime();
    vector<int> roots;
    for(int i=0;i<trees;++i)
    {
        ProjectX(i);
        ProjectY(i);
        compute_magnitudes_train();
        copy_projections();
        tbeg = MPI_Wtime();
        roots.push_back(train_tree(instances,0));
        elapsedTime = MPI_Wtime() - tbeg;
        train_time += elapsedTime;
        cout << train_time << endl;
    }

    double maxTime;
    MPI_Reduce( &train_time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
    if ( rank == 0 ) {
        printf( "total duration of training :%f\n", maxTime);
    }

    meanvector.resize(ydim);
    globalmeanvector.resize(ydim);
    cmpvector.resize(ydim);
    Mgt.resize(Nt);

    MPI_Barrier( MPI_COMM_WORLD );
    unordered_map<int,double> tp;
    for(int i=0;i<Nt;++i)
    {
        Xpt.push_back(tp);
        Ypt.push_back(tp);
    }

    for(int i=0;i<Nt;++i)
    {
        fill(meanvector.begin(),meanvector.end(),0.0);
        // cout << "classifying test " << i << endl;
        for(int j=0;j<trees;++j)
        {
            ProjectXt(j,i);
            ProjectYt(j,i);
            compute_magnitudes_test(i);
            test(roots[j],i);
        }
        
        MPI_Reduce(meanvector.data(),globalmeanvector.data(),ydim,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(rank == 0)
        {
            for(int j=0;j<ydim;++j)
                cmpvector[j] = {globalmeanvector[j],j};
            sort(cmpvector.begin(),cmpvector.end());
            reverse(cmpvector.begin(),cmpvector.end());
            for(int l=1;l<=5;l+=2)
            {
                for(int j=0;j<min(l,(int)cmpvector.size());++j)
                {
                    if(Yt[i][cmpvector[j].second])
                        p[l] += (1.0)/(double)(l);
                }
            }
        }
    }

    if(rank == 0)
    {
        for(int i=1;i<=5;i+=2)
            cout << "p@" << i << " : " << 100.0*(p[i]/Nt) << "\n"; 
    }

    /* shut down MPI */
    MPI_Finalize();
    return 0;
}
