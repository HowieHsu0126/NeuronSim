project(ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275)

# Targets  
add_executable(ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275
    src/main_ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275.cpp
)

# definitions     
if(CARLSIM_NO_CUDA)
    target_compile_definitions(ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275 PUBLIC
        -D__NO_CUDA__ )
endif()

# Linking    
target_link_libraries(ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275 PRIVATE
    carlsim
    carlsim-stopwatch
)
target_include_directories(ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275 PRIVATE
    ../../tools/stopwatch
)
   
        
# Installation   
install(TARGETS ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275
    EXPORT carlsim
    RUNTIME 
        DESTINATION $<$<CONFIG:Debug>:debug/>samples 
        COMPONENT ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275
        )

#install(FILES ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275 DESTINATION $<$<CONFIG:Debug>:debug/>samples)          