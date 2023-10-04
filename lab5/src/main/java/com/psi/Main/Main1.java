package com.psi.Main;

import java.util.Date;

import static com.psi.Solution.zad1;

public class Main1 {

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        try {
            zad1();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("Time: " + (System.currentTimeMillis() - startTime) / 1000.0);
        System.out.println(new Date());

    }
}
