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
   
10) Randomly select k elements from a list a:

   ```{.isa}
   a) random.shuffle(a); a[:k]
   b) random.sample(a, k)
   c) If the randomly selected set need to be the same in each iteration: random.seed(l); random.sample(a, k)
   ```
   
### Array



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
   
3） Set columns of a dataframe:

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

### Keras package for neual network


### Some useful tricks

1) Measure the running time:

   ```{.isa}
   import time
   start = time.time()
   (function running here...)
   end = time.time()
   running_time = end - start
