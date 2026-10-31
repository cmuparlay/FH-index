foreach(UTILITY IN ITEMS generate_dataset sort_dataset)
    fh_index_add_program(${UTILITY}_lat1_typeNONE_b0_f0 "${UTILITY}.cpp")
endforeach()
