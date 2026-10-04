add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

# Fork-local server launchers. start-bonsai-server.bat / start-qwen38-server.bat resolve
# `%~dp0ninfer-serve.exe`, so stage them next to the server binary after every build. They are
# fork conveniences that may be absent (e.g. a clean upstream checkout), so guard on existence
# instead of failing configure when they are missing.
foreach(ninfer_launcher
    "${PROJECT_SOURCE_DIR}/start-bonsai-server.bat"
    "${PROJECT_SOURCE_DIR}/start-qwen38-server.bat")
  if(EXISTS "${ninfer_launcher}")
    add_custom_command(TARGET ninfer-serve POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${ninfer_launcher}"
              "$<TARGET_FILE_DIR:ninfer-serve>"
      COMMENT "Staging ${ninfer_launcher} next to the server binary")
  endif()
endforeach()

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
