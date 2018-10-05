####################################

## Some frequently used commands

####################################

### List
1) Convert a binary feature in dataframe to one-hot:

   ```{.isa}
   a = df.loc[:,'feature'].values.tolist();  pd.get_dummies(a)
   ```

2) Find a certain element x in list a:

   ```{.isa}
   a) a.index(x) -- Only return the first element
   b) [i for i, j in enumerate(a) if j == x]
   ```

3) Delete an element x from list a:

   ```{.isa}
   a) a.remove(x)  -- Only delete the first element
   b) del a[x]
   c) a.pop(x) -- Could return the value being popped
   ```
   
4) Delete a list of elements b from another list a:

   ```{.isa}
   a) [x for x in b if x not in a]
   b) c = sorted(set(a) - set(b))
   ```

5) About SET:
   
   ```{.isa}
   a) Get unique elements in a list a: set(a) 
   b) Remove a list b from list a: list(set(a) - set(b))
   c) Intersection of two sets a and b: set(a) & set(b)
   d) Union of two sets a and b: set(a) | set(b)
   ```
   
6) Iteratively generate a dictionary:

   ```{.isa}
   {i: A[i] for i in range(len(A))}
   ```
   
   Add an element to dictionary:
   
   ```{.isa}
   a) A_dict['x'] = a
   b) A_dict.update{'x': a}
   ```
   
7) Sort a list a:

   ```{.isa}
   a) a.sort()
   b) a.sort(reverse=True) (with decresing order) 
   c) sorted(range(len(a)), key=lamba k:a[k]) (return the index of resorted data)
   ```
   
8) Divide a number for each element in list a:

   ```{.isa}
   [x/N for x in a]
   ```
 
9) Subtract two lists a and b:

   ```{.isa}
   [x1 - x2 for (x1, x2) in zip(a,b)]
   list(np.array(a) - np.array(b))
   ```
   
10) Usage of **zip**:
   
   Return a tuple that aggregates elements from each of the iterables:
   
   ```{.isa}
   zipped = zip(x,y)
   zipped = list(zipped)
   ```
   Note: x and y can have different lengths, the generated tuple has the same length with the shorter list. 
   
   Unzip a zipped tuple:
   
   ```{.isa}
   x1, y1 = zip(*zipped)
   ```
   
11) Randomly select k elements from a list a:

   ```{.isa}
   a) random.shuffle(a); a[:k]
   b) random.sample(a, k)
   c) If the randomly selected set need to be the same in each iteration: random.seed(l); random.sample(a, k)
   ```
   
12) Usage of **MAP**:
   
   Applies funtion to all items in an input list, i.e. square each element in a list a:
   
   ```{.isa}
   list(map(lambda x: x**2, a))
   ```
   Work with a input list of functions (Apply functions as inputs):
   
   ```{.isa}
   def Multiply(x): return x*x
   def Add(x): return x+x
   funcs = [Multiply, Add]
   for i in range(5):
      val = list(map(lambda x: x(i), funcs))
   ```
   
13) Usage of **Filter**：

   Creat a sublist of elements meet with the condition, i.e.: 

   ```{.isa}
   list(filter(lambda x: x < 0, input_list))
   ```
   
14) Usage of **Reduce**:

   A rolling computation to sequential pairs of values in a list, i.e.:
   
   ```{.isa}
   from functools import reduce
   product = reduce((lambda x,y: x*y), [1,2,3,4])
   Output: 24
   ```

### Array

1) Concatenate two arrays A and B:
   
   ```{.isa}
   np.concatenate((A, B), axis = 0/1)
   Note: axis = 0 -- concatenate all elements in A and B into a column of array
         axis = 1 -- concatenate pairwise elements in A and B with the same size of A/B
   ```
   
2) Index of an element a in array A:
   
   ```{.isa}
   index = np.where(A == a)[0] 
   ```
   
   np.where to select elements from two lists(If True, select elements from first list; else select from second list):
   ```{.isa}
   np.where([True, False],[True, True]], [[1,2],[3,4]], [[5,6],[7,8]]) --> array([[1,6],[3,4]])
   ```
   
3) Generate an array A with elements from a to b:

   ```{.isa}
   A = np.arange(a,b)
   ```
   
4) Find unique elements in an array A:

   ```{.isa}
   [unique_elements, idx, counts] = np.unique(A, return_index = True, return_counts = True)
   ```

5) Find unique rows in an array A (use `void` type to jion the whole row into a single item):

   ```{.isa}
   B = np.ascontiguousarray(A).view(np.dtype((np.void, A.dtype.itemsize * A.shape[1])))
   [_, idx] = np.unique(B, return_index = True)
   unique_A = A[idx]
   ```
   Note: `ascontiguousarray` return a contiguous array in memory, can be checked by operator `B.flags['C_CONTIGUOUS']`
   
6) Cummulative sum of an array A:

   a) Cummulative sum of all elements into a list: `np.cumsum(A)`
   
   b) Cummulative sum of elements by row/column, returned a matrix with the same size: `np.cumsum(A, axis = 0/1)`
   
7) Calcualte sums within an array A: 
   
   a) Sum of all elements: `np.sum(A)`
   
   b) Sum up rows/columns: `np.sum(A, axis=0/1)`
   
8) Calcualte difference of adjacent elements along the given axis in an array A:

   ```{.isa}
   bp.diff(A)
   ```

### DataFrame

1) Append dataframes iteratively: 

   ```{.isa}
   comb_df = []
   for i in range(n):
      comb_df.append(sub_df[n]) 
   comb_df = pd.concat(comb_df, axis=1)
   ```

2) Determine the maximum value column in each row of dataframe:

   ```{.isa}
   df.apply(lamba x: x.argmax(), axis=1)
   (df.apply: Apply function along input axis of dataframe)
   ```
   
3) Set columns of a dataframe:

   ```{.isa}
   df.columns = [list of column names]
   ```
   
4) Select dataframe rows by a value / a list of values:

   ```{.isa}
   df[df['A'] == n]
   df[df['A'].isin([N])]
   ```

5) Select dataframe columns by a list of names:
   
   ```{.isa}
   df.loc[:, ['A']]
   df.loc[:, [N]]
   ```
   
6) Reset indexes in a dataframe (without keeping the previous indexes):

   ```{.isa}
   df.reset_index(drop=True)
   ```
   
7) Sort a dataframe by a set of columns:

   ```{.isa}
   df.sort_values(by=[N])
   ```
   
8) Get the null indexes of a column in dataframe:

   ```{.isa}
   df[df['A'].isnull()].index
   sum(df['A'].isnull) (Number of null elements in a column)
   ```
9) Row index of a dataframe for NaN in a column 'A':

   ```{.isa}
   np.where(pd.isnull(df['A']))[0]
   ```

10) Generate a dataframe with all elements being 0:

   ```{.isa}
   pd.DataFrame(0, index=np.arange(n), columns=[columns_set])
   ```
   
11) Rename columns in dataframe:

   ```{.isa}
   df.rename(index=str, columns = {"A": "a"; "B": "b"})
   ```
  
12) Find out unique elements and their numbers in a column: 

   ```{.isa}
   selected_demg_df.groupby('column')['ID'].nunique()
   ```

### Keras package for neual network


### Some useful tricks

1) Measure the running time:

   ```{.isa}
   import time
   start = time.time()
   (function running here...)
   end = time.time()
   running_time = end - start
   ```
   
2) Calculate e^x:

   ```{.isa}
   import math
   math.exp(x)
   ```
3) Calculate n^a:

   ```{.isa}
   n**(a)
   ```
4) np.array v.s np.asarray:

   `array` will make a copy of the object, while `asarray` will not unless necessary. 
   
5) Tuple v.s. List:

   a) Tuple is fixed size whereas list is dynamic. i.e. elements cannot be added, removed from tuple. The operator `in` can be utilzied to check if an element exits in tuple.
   
   b) Tuple can be utilized as dictionary keys. 
   
   c) Tuple utilized for hererogeneous collections, while list for homogeneous collections.
   
6) Shallow copy v.s Deep copy:

   a) Shallow copy constructs a new compound object, and the inserts references into it to the objects found in the original;
   
   b) Deep copy also constructs a new compound object, and the inserts references to the copied objects in the new one.
   
7) list.append v.s list.extend:

   a) `append`: A = [a]; A.append([b]) --> [a,[b]]
   
   b) `extend`: A = [a]; A.extend([b]) --> [a,b]
   
8) Get all possible combination of the elements in a list A:

   ```{.isa}
   import itertools
   itertools.combinations(A)
   ```
   
   Get all possible combination of different lists in A (with one element selected from each list):
   
   ```{.isa}
   import itertools
   list(itertools.product(*A))
   ```
9) Partition data into training and testing by `from sklearn.cross_validation import train_test_split`   
      
   Parition data into training, validation and testing sets:  
   ```  
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
   ```
10) K-fold cross validation: `from sklearn.cross_validation import KFold`
   
   ```{.isa}
   kf = KFold(n_splits=10)
   for train, test in kf.split(dataset):
      trainFrame = dataset.iloc[train]
      testFrame = dataset.iloc[test]
   ```
11) Calcualte pairwaise euclidean distance: 
```
import scipy
scipy.spatial.distance.cdist(df.iloc[:,1:], df.iloc[:,1:], metric='eculidean')
```
