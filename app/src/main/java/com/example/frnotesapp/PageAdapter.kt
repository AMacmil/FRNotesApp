package com.example.frnotesapp

import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.viewpager2.adapter.FragmentStateAdapter

// PageAdapter for managing fragments in ViewPager2
class PageAdapter(fa: AppCompatActivity, private val mNumOfTabs: Int) :
    FragmentStateAdapter(fa) {

    override fun getItemCount(): Int {
        return mNumOfTabs  // return number of tabs
    }

    // return the Fragment associated with the specified position (chosen tab)
    override fun createFragment(position: Int): Fragment {
        return when (position) {
            0 -> HomeFragment()
            1 -> NotesFragment()
            2 -> LogsFragment()
            else -> HomeFragment()
        }
}
    }