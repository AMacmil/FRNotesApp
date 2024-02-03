package com.example.frnotesapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatDelegate
import androidx.appcompat.widget.Toolbar
import androidx.lifecycle.ViewModelProvider
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import kotlin.properties.Delegates

class MainActivity : AppCompatActivity()  {
    private lateinit var viewModel: NotesViewModel
    private lateinit var tabLayout: TabLayout

    // isAuthenticated is an observable property - its value determines the visibility of tab navigation
    var isAuthenticated by Delegates.observable(false) { _, _, newValue ->
        if(::tabLayout.isInitialized) {
            updateTabLayoutVisibility(newValue)
        }
    }

    // onCreate method is called when the activity is first created.
    override fun onCreate(savedInstanceState: Bundle?) {

        // apply theme based on dark mode setting - onCreate is called if the phone is
        // switched to dark mode so it's appropriate here
        when (AppCompatDelegate.getDefaultNightMode()) {
            AppCompatDelegate.MODE_NIGHT_YES,
            AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM,
            AppCompatDelegate.MODE_NIGHT_UNSPECIFIED -> {
                setTheme(R.style.Base_Theme_FRNotesApp)
            }
            // else, apply light theme
            AppCompatDelegate.MODE_NIGHT_NO -> {
                setTheme(R.style.Base_Theme_FRNotesApp)
            }
        }

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // initialize ViewModel, set up db connection and get existing data
        viewModel = ViewModelProvider(this)[NotesViewModel::class.java]
        viewModel.initDatabase(this)
        viewModel.updateNoteListFromDatabase()

        // set toolbar as the app's action bar
        val toolbar = findViewById<Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)

        if (supportActionBar != null) {
            supportActionBar?.title = "";
        }

        // initialize TabLayout and add titled tabs
        tabLayout = findViewById<TabLayout>(R.id.tab_layout)
        tabLayout.addTab(tabLayout.newTab().setText(R.string.tab_page1))
        tabLayout.addTab(tabLayout.newTab().setText(R.string.tab_page2))
        tabLayout.addTab(tabLayout.newTab().setText(R.string.tab_page3))

        // initialize ViewPager2 and set its adapter - the PageAdaptor defines how the
        // Fragments are linked to the tabs
        val viewPager = findViewById<ViewPager2>(R.id.pager)
        viewPager.isUserInputEnabled = false
        val adapter = PageAdapter(this, 3)
        viewPager.adapter = adapter

        // attach TabLayout to ViewPager2
        // TabLayoutMediator links ViewPager2 and TabLayout &  allows switching tabs
        TabLayoutMediator(tabLayout, viewPager) { tab, position ->
            // assign text to each tab based on position.
            when (position) {
                0 -> tab.text = "Home"
                1 -> tab.text = "Notes"
                2 -> tab.text = "Logs"
            }
        }.attach()

        updateTabLayoutVisibility(isAuthenticated)
    }// end onCreate

    private fun updateTabLayoutVisibility(isAuthenticated: Boolean) {
        tabLayout.visibility = if (isAuthenticated) View.VISIBLE else View.GONE
    }

}// end class MainActivity