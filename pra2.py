class Solution:
    def firstPalindrome():
        l=["abc","car","ada","racecar","cool"]
        c=0
        for a in l:
            if a.lower()==a[::-1]:
                c+=1
                print(a)
                break
    firstPalindrome()