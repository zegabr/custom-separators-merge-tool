StepVerifier.create(result).expectErrorSatisfies((exception) -> {
  if (!(exception instanceof SSLException)) {
    assertThat(exception).hasCauseInstanceOf(WebClientRequestException.class);
  }
}).verify(Duration.ofSeconds(10));